use super::*;
use sha2::{Digest, Sha256};

fn npy(values: &[f32], dtype: &str) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    header.push_str(&" ".repeat(64 - (11 + header.len()) % 64));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.bytes());
    out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    out
}

fn upload(
    app: &Fixture,
    model_id: &str,
    manifest: &Value,
    vector: &[u8],
    auth: &str,
) -> (u16, Value) {
    let boundary = "h4i-unit-vector";
    let mut body = Vec::new();
    for (name, value) in [("model_id", model_id), ("manifest", &manifest.to_string())] {
        write!(
            &mut body,
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .unwrap();
    }
    write!(&mut body, "--{boundary}\r\nContent-Disposition: form-data; name=\"vector\"; filename=\"unit.npy\"\r\n\r\n").unwrap();
    body.extend(vector);
    write!(&mut body, "\r\n--{boundary}--\r\n").unwrap();
    let mut conn = app.open_request(
        "POST",
        "/api/admin/directions/import",
        None,
        &format!("{auth}Content-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\n", body.len()),
    );
    conn.write_all(&body).unwrap();
    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    let (head, content) = response.split_once("\r\n\r\n").unwrap();
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        serde_json::from_str(content).unwrap_or_else(|_| json!(content)),
    )
}

#[test]
fn operator_imports_verified_v2_direction_and_agents_can_list_read_and_compare() {
    let mut app = Fixture::new();
    let credentials = app.path("direction-operator.json");
    let token = "a".repeat(64);
    fs::write(
        &credentials,
        json!({"operator":"test-operator","token":token}).to_string(),
    )
    .unwrap();
    fs::set_permissions(&credentials, fs::Permissions::from_mode(0o600)).unwrap();
    app.launch_with(&[("BREWERY_DIRECTION_OPERATOR_FILE", credentials)]);
    let model = app.register("alpha", "hgn-v2", "/models/import-fixture.hgn");
    let v = npy(&[1., 0., 0.], "<f4");
    let manifest_for = |v: &[u8]| {
        json!({
            "layer":1,"position":-1,"model_sha256":"d".repeat(64),
            "npy_sha256":format!("{:x}",Sha256::digest(v)), "provenance":"Synthetic axis"
        })
    };
    let manifest = manifest_for(&v);
    let call = |manifest: &Value, v: &[u8], auth: &str| upload(&app, &model, manifest, v, auth);
    let auth = format!("Authorization: Bearer {token}\r\n");
    assert_eq!(call(&manifest, &v, "").0, 403);
    assert_eq!(
        call(&manifest, &v, "Authorization: Bearer wrong\r\n").0,
        403
    );
    for (field, value) in [
        ("npy_sha256", json!("b".repeat(64))),
        ("model_sha256", json!("b".repeat(64))),
        ("layer", json!(2)),
    ] {
        let mut wrong = manifest.clone();
        wrong[field] = value;
        let reply = call(&wrong, &v, &auth);
        assert_eq!(reply.0, 400, "{field}: {reply:?}");
    }
    for (values, dtype) in [
        (&[1., 0.][..], "<f4"),
        (&[1.0002, 0., 0.][..], "<f4"),
        (&[1., f32::NAN, 0.][..], "<f4"),
        (&[1., 0., 0.][..], "<i4"),
    ] {
        let bad = npy(values, dtype);
        let reply = call(&manifest_for(&bad), &bad, &auth);
        assert_eq!(reply.0, 400, "{reply:?}");
    }
    assert_eq!(
        app.json("GET", "/api/directions", None, 200)["directions"],
        json!([])
    );
    let (status, imported) = call(&manifest, &v, &auth);
    assert_eq!(status, 201, "{imported}");
    let id = imported["id"].as_str().unwrap();
    let path = format!("/api/directions/{id}/versions/2");
    let artifact = app.json("GET", &path, None, 200);
    assert_eq!(artifact["direction"]["origin"], "imported");
    assert_eq!(artifact["direction"]["import"]["manifest"], manifest);
    assert_eq!(artifact["direction"]["import"]["operator"], "test-operator");
    assert!(
        artifact["direction"]["import"]["imported_at_ms"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        artifact["direction"]["fit"]["layers"][0]["unit_vector"],
        json!([1., 0., 0.])
    );
    let listed = app.json("GET", "/api/directions", None, 200);
    assert_eq!(listed["directions"][0]["id"], id);
    assert_eq!(listed["directions"][0]["origin"], "imported");
    assert!(listed["directions"][0].get("fit").is_none());
    assert!(listed["directions"][0].get("import").is_none());
    let mcp =
        |name: &str| {
            let (status, body) = app.request("POST", "/mcp", Some(json!({
            "jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":{}}
        })), "Accept: application/json, text/event-stream\r\n");
            assert_eq!(status, 200, "{body}");
            serde_json::from_str::<Value>(&body).unwrap()
        };
    assert_eq!(
        mcp("list_directions")["result"]["structuredContent"],
        listed
    );
    assert_eq!(mcp("import_direction")["error"]["message"], "unknown tool");
    assert!(
        app.request(
            "POST",
            "/api/jobs",
            Some(json!({
                "kind":"imported_direction","input_ids":[model]
            })),
            ""
        )
        .0 >= 400
    );
    let compared = app.json(
        "GET",
        &format!("{path}?layer=1&reference_id={id}&reference_layer=1"),
        None,
        200,
    );
    assert_eq!(
        compared["reference"]["cosine_by_layer"],
        json!([{"layer":1,"cosine":1.}])
    );
}
