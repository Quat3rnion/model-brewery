use super::*;

fn fixture(t: &Test, values: &[f32]) -> std::path::PathBuf {
    let dir = t.root.path().join("scratch-fixtures");
    fs::create_dir(&dir).unwrap();
    let path = dir.join("parent.safetensors");
    // Include metadata and trailing header padding to check byte-exact restoration.
    let header = br#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}        "#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    fs::write(&path, bytes).unwrap();
    let config = t.root.path().join("scratch-worker.json");
    fs::write(
        &config,
        json!({"source_roots":[dir],"disk_reserve_bytes":0}).to_string(),
    )
    .unwrap();
    fs::write(
        t.root.path().join("dispatch.json"),
        json!({"hosts":{"existing":{"local":true,"config":config}}}).to_string(),
    )
    .unwrap();
    fs::remove_file(t.root.path().join("catalog.json")).unwrap();
    for (name, script) in [
        ("editor-DISPATCH", "#!/bin/sh\nexit 1\n"),
        ("editor-WORKER", "#!/bin/sh\nexit 1\n"),
        ("editor-ENGINE", "#!/bin/sh\nexit 1\n"),
        (
            "editor-HELPER",
            r#"#!/bin/sh
test "$1" = --capabilities || exit 1
printf '%s\n' '{"schema":"brewery.weight-runtime.v1","jobs":["weight_inspect","weight_propose","weight_apply"],"formats":["gguf","hf","safetensors"],"visual":true,"packed":false}'
"#,
        ),
    ] {
        let file = t.root.path().join(name);
        fs::write(&file, script).unwrap();
        fs::set_permissions(file, fs::Permissions::from_mode(0o700)).unwrap();
    }
    path
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn scratch_edits_keep_parents_and_require_separate_publish_approval() {
    let mut t = Test::new();
    let source = fixture(&t, &[1., 2., 3., 4., 5., 6.]);
    let original = fs::read(&source).unwrap();
    let parent_hash = sha(&original);
    t.launch();
    let (status, parent) = t.request(
        "/api/models",
        "",
        Some(json!({"host":"existing","kind":"safetensors","path":source})),
    );
    assert_eq!(status, 201, "{parent}");
    let create = json!({"parent_id":parent["id"],"parent_sha256":parent_hash});
    let path = "/api/scratch";
    assert_eq!(t.request(path, "", Some(create.clone())).0, 409);
    assert_eq!(
        t.admin(Some(
            json!({"action":"set_weight_editor","worker":"existing",
            "revision":0,"enabled":true})
        ))
        .0,
        200
    );
    let mut invalid = create.clone();
    invalid["parent_sha256"] = json!("0".repeat(64));
    assert_eq!(t.request(path, "", Some(invalid.clone())).0, 400);
    let key = "Idempotency-Key: scratch.create\r\n";
    let (status, scratch) = t.request(path, key, Some(create.clone()));
    assert_eq!(status, 201, "{scratch}");
    assert_eq!(scratch["kind"], "scratch");
    assert_eq!(scratch["sha256"], parent_hash);
    assert_eq!(t.request(path, key, Some(create)), (200, scratch.clone()));
    assert_eq!(t.request(path, key, Some(invalid)).0, 409);

    let id = scratch["id"].as_str().unwrap();
    let root = t.root.path().join("state/scratch");
    let model = |id: &str| root.join(id).join("model.safetensors");
    assert_eq!(fs::read(model(id)).unwrap(), original);
    let edit = format!("{path}/{id}/edits");
    let mut request = json!({"operations":[
        {"op":"set","tensor":"w","origin":[0,1],"shape":[1,2],"values":[8.,9.]},
        {"op":"add","tensor":"w","origin":[1,0],"shape":[1,2],"values":[1.,2.]},
        {"op":"low_rank","tensor":"w","u":[2.,1.],"v":[3.,0.,2.],"scale":0.5},
        {"op":"orthogonalize","tensor":"w","axis":"rows","direction":[1.,0.,0.]}
    ]});
    request["operations"][3]["direction"] = json!([0., 0., 0.]);
    assert_eq!(t.request(&edit, "", Some(request.clone())).0, 400);
    request["operations"][3]["direction"] = json!([1., 0., 0.]);
    let key = "Idempotency-Key: scratch.edit\r\n";
    let (status, child) = t.request(&edit, key, Some(request.clone()));
    assert_eq!(status, 201, "{child}");
    assert_eq!(t.request(&edit, key, Some(request)), (200, child.clone()));
    assert_eq!(child["provenance"]["parent_id"], id);
    assert_eq!(child["provenance"]["parent_sha256"], parent_hash);
    assert_eq!(
        child["provenance"]["tensors"],
        json!([{"name":"w","shape":[2,3],"dtype":"F32"}])
    );
    assert_eq!(child["qualification"], "not release-qualified");
    assert_eq!(sha(&fs::read(&source).unwrap()), parent_hash);
    assert_eq!(sha(&fs::read(model(id)).unwrap()), parent_hash);
    let child_id = child["id"].as_str().unwrap();
    let output = fs::read(model(child_id)).unwrap();
    let values: Vec<_> = output[output.len() - 24..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(values, [0., 8., 11., 0., 7., 7.]);
    assert_eq!(sha(&output), child["sha256"]);
    let delta_file = root.join(child_id).join("delta.bin");
    let restored = t.root.path().join("reverted.safetensors");
    brewery_worker::weight_source::scratch::revert(&model(child_id), &delta_file, &restored)
        .unwrap();
    assert_eq!(fs::read(&restored).unwrap(), original);
    assert!(
        brewery_worker::weight_source::scratch::revert(&model(child_id), &delta_file, &source)
            .is_err()
    );
    assert_eq!(sha(&fs::read(&source).unwrap()), parent_hash);
    let corrupt = t.root.path().join("corrupt.bin");
    let mut damaged = fs::read(&delta_file).unwrap();
    *damaged.last_mut().unwrap() ^= 1;
    fs::write(&corrupt, damaged).unwrap();
    let rejected = t.root.path().join("rejected.safetensors");
    assert!(
        brewery_worker::weight_source::scratch::revert(&model(child_id), &corrupt, &rejected)
            .is_err()
    );
    assert!(!rejected.exists());
    let summary = format!("{path}/{child_id}/delta");
    assert_eq!(t.request(&summary, "", None).1["changed_tensors"], 1);
    assert_eq!(
        t.request("/api/state", "", None).1["models"],
        json!([parent])
    );
    let alias = json!({"host":"existing","kind":"safetensors","path":model(child_id)});
    assert_eq!(t.request("/api/models", "", Some(alias)).0, 400);
    let link = t.root.path().join("alias.safetensors");
    std::os::unix::fs::symlink(model(child_id), &link).unwrap();
    assert_eq!(
        t.request(
            "/api/models",
            "",
            Some(json!({"host":"existing","kind":"safetensors","path":link}))
        )
        .0,
        400
    );
    let approval = format!("/api/admin/scratch/{child_id}/approve");
    let review = json!({"sha256":child["sha256"]});
    assert_eq!(t.request(&approval, "", Some(review.clone())).0, 409);
    assert_eq!(
        t.request(&format!("{path}/{id}/discard"), "", Some(json!({})))
            .0,
        409
    );
    let requested = format!("{path}/{child_id}/publish-request");
    assert_eq!(
        t.request(&requested, "", Some(review.clone())).1["state"],
        "publish_requested"
    );
    assert_eq!(
        t.request(
            &approval,
            "Origin: https://other.invalid\r\n",
            Some(review.clone())
        )
        .0,
        403
    );
    t.stop();
    t.launch();
    assert_eq!(t.request(path, "", None).1.as_array().unwrap().len(), 2);
    let (status, approved) = t.request(&approval, "Idempotency-Key: approve\r\n", Some(review));
    assert_eq!(status, 200, "{approved}");
    assert_eq!(approved["state"], "approved");
    let library = t.request("/api/state", "", None).1["models"].clone();
    assert!(
        library
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == approved["library_model_id"] && m["kind"] == "safetensors")
    );

    let (_, leaf) = t.request(
        path,
        "",
        Some(json!({"parent_id":child_id,"parent_sha256":child["sha256"]})),
    );
    let leaf_id = leaf["id"].as_str().unwrap();
    assert!(model(leaf_id).is_file());
    let discard = format!("{path}/{leaf_id}/discard");
    let key = "Idempotency-Key: scratch.discard\r\n";
    let (status, discarded) = t.request(&discard, key, Some(json!({})));
    assert_eq!(status, 200, "{discarded}");
    assert_eq!(t.request(&discard, key, Some(json!({}))), (200, discarded));
    assert!(!model(leaf_id).exists());
    assert_eq!(
        t.request(&format!("{path}/{leaf_id}/delta"), "", None).0,
        404
    );
    assert_eq!(t.request(path, "", None).1.as_array().unwrap().len(), 2);
    assert_eq!(fs::read(model(child_id)).unwrap(), output);
}
