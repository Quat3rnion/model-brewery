use super::*;

fn fixture(dir: &Path, name: &str, values: &[f32]) -> (std::path::PathBuf, String) {
    let json = format!(
        r#"{{"weight":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        values.len(),
        values.len() * 4
    );
    let mut data = (json.len() as u64).to_le_bytes().to_vec();
    data.extend(json.bytes());
    data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    let path = dir.join(format!("{name}.safetensors"));
    std::fs::write(&path, &data).unwrap();
    (path, format!("{:x}", Sha256::digest(data)))
}

#[test]
fn task_arithmetic_writes_reversible_child() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let (base, pin) = fixture(&root, "base", &[1., 2., 3., 4.]);
    let (a, a_pin) = fixture(&root, "a", &[5., 2., 1., 4.]);
    let (b, b_pin) = fixture(&root, "b", &[3., 3., 3., 4.]);
    let inputs = [
        Input {
            path: &a,
            sha256: &a_pin,
            weight: 0.5,
        },
        Input {
            path: &b,
            sha256: &b_pin,
            weight: 2.,
        },
    ];
    let result = build(&base, &pin, &root, Method::TaskArithmetic, &inputs).unwrap();
    let (_, expected) = fixture(&root, "expected", &[7., 4., 2., 4.]);
    assert_eq!(result.sha256, expected);
    assert_eq!(result.delta.changed_tensors, 1);
    let restored = root.join("restored.safetensors");
    revert(&root.join(MODEL), &root.join(DELTA), &restored).unwrap();
    assert_eq!(
        std::fs::read(restored).unwrap(),
        std::fs::read(base).unwrap()
    );
}

#[test]
fn matches_pinned_mergekit_outputs() {
    #[derive(Deserialize)]
    struct Reference {
        inputs: [Vec<f32>; 3],
        outputs: BTreeMap<String, Expected>,
    }
    #[derive(Deserialize)]
    struct Expected {
        values: Vec<f32>,
        sha256_f32le: String,
    }
    let r: Reference = serde_json::from_str(include_str!("reference.json")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let [(base, pin), (a, ap), (b, bp)] =
        std::array::from_fn(|i| fixture(&root, &i.to_string(), &r.inputs[i]));
    let inputs = [
        Input {
            path: &a,
            sha256: &ap,
            weight: 0.5,
        },
        Input {
            path: &b,
            sha256: &bp,
            weight: 2.,
        },
    ];
    let cases = [
        Method::Linear,
        Method::TaskArithmetic,
        Method::Ties { density: 0.4 },
        Method::DareLinear {
            drop_rate: 0.,
            seed: 2026,
        },
        Method::DareTies {
            drop_rate: 0.,
            seed: 2026,
        },
    ];
    for method in cases {
        let name = method.name();
        let out = root.join(name);
        std::fs::create_dir(&out).unwrap();
        build(&base, &pin, &out, method, &inputs).unwrap();
        let expected = &r.outputs[name];
        let bytes = std::fs::read(out.join(MODEL)).unwrap();
        let start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(bytes.len() - start, expected.values.len() * 4);
        let actual = &bytes[start..];
        let expected_bytes: Vec<u8> = expected
            .values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(&expected_bytes)),
            expected.sha256_f32le
        );
        for (bytes, &target) in actual.chunks_exact(4).zip(&expected.values) {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            assert!(
                (value - target).abs() <= 1e-6,
                "{name}: {value} != {target}"
            );
        }
        if name.starts_with("dare") {
            assert_eq!(actual, expected_bytes, "{name} at p=0 must match exactly");
        }
    }
}

#[test]
fn dare_rescales_survivors_and_has_seeded_replay() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let n = 100_000;
    let (base, pin) = fixture(&root, "base", &vec![0.; n]);
    let (a, ap) = fixture(&root, "a", &vec![3.; n]);
    let inputs = [Input {
        path: &a,
        sha256: &ap,
        weight: 1.,
    }];
    for method in [
        |seed| Method::DareLinear {
            drop_rate: 0.75,
            seed,
        },
        |seed| Method::DareTies {
            drop_rate: 0.75,
            seed,
        },
    ] {
        let run = |label, seed| {
            let out = root.join(label);
            std::fs::create_dir(&out).unwrap();
            let result = build(&base, &pin, &out, method(seed), &inputs).unwrap();
            (out, result)
        };
        let (out, x) = run("first", 123);
        let (_, replay) = run("replay", 123);
        let (_, different) = run("different", 124);
        assert_eq!(x.sha256, replay.sha256);
        assert_eq!(x.delta.delta_sha256, replay.delta.delta_sha256);
        assert_ne!(x.sha256, different.sha256);
        let bytes = std::fs::read(out.join(MODEL)).unwrap();
        let mut survivors = 0;
        for value in bytes[bytes.len() - n * 4..].chunks_exact(4) {
            match f32::from_le_bytes(value.try_into().unwrap()) {
                0. => (),
                12. => survivors += 1,
                other => panic!("rescale should yield 12, got {other}"),
            }
        }
        assert!(
            (24_500..=25_500).contains(&survivors),
            "{survivors} survivors"
        );
        for dir in ["first", "replay", "different"] {
            std::fs::remove_dir_all(root.join(dir)).unwrap();
        }
    }
}

#[test]
fn ties_breaks_magnitude_ties_at_the_first_flat_index() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let (base, pin) = fixture(&root, "base", &[1., 0., 0., 0.]);
    let (a, ap) = fixture(&root, "a", &[3., -2., 3., 0.]);
    let inputs = [Input {
        path: &a,
        sha256: &ap,
        weight: 2.,
    }];
    let result = build(&base, &pin, &root, Method::Ties { density: 0.5 }, &inputs).unwrap();
    let (_, expected) = fixture(&root, "expected", &[3., 0., 3., 0.]);
    assert_eq!(result.sha256, expected);
}

#[test]
fn ties_elects_positive_on_exact_sign_tie() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let (base, pin) = fixture(&root, "base", &[0.]);
    let (a, ap) = fixture(&root, "a", &[3.]);
    let (b, bp) = fixture(&root, "b", &[-3.]);
    let inputs = [
        Input {
            path: &a,
            sha256: &ap,
            weight: 1.,
        },
        Input {
            path: &b,
            sha256: &bp,
            weight: 1.,
        },
    ];
    let result = build(&base, &pin, &root, Method::Ties { density: 1. }, &inputs).unwrap();
    assert_eq!(result.sha256, ap);
}

#[test]
fn diff_and_revert_read_more_than_32_merge_records() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let header = serde_json::to_vec(
        &(0..40)
            .map(|i| {
                (
                    format!("w{i:02}"),
                    serde_json::json!({
                        "dtype":"F32","shape":[1],"data_offsets":[i * 4, i * 4 + 4]
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap();
    let write = |name, value: f32| {
        let path = root.join(format!("{name}.safetensors"));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(&header);
        bytes.extend(value.to_le_bytes().repeat(40));
        std::fs::write(&path, &bytes).unwrap();
        (path, format!("{:x}", Sha256::digest(bytes)))
    };
    let (base, pin) = write("base", 1.);
    let (peer, peer_pin) = write("peer", 2.);
    let inputs = [Input {
        path: &peer,
        sha256: &peer_pin,
        weight: 1.,
    }];
    let result = build(&base, &pin, &root, Method::TaskArithmetic, &inputs).unwrap();
    assert_eq!(result.delta.changed_tensors, 40);
    let rows = diff::compare(
        &mut File::open(&base).unwrap(),
        &pin,
        &mut File::open(root.join(MODEL)).unwrap(),
        &result.sha256,
        Some(diff::Delta {
            file: &mut File::open(root.join(DELTA)).unwrap(),
            summary: &result.delta,
        }),
    )
    .unwrap();
    assert_eq!(rows.len(), 40);
    let restored = root.join("restored.safetensors");
    revert(&root.join(MODEL), &root.join(DELTA), &restored).unwrap();
    assert_eq!(
        std::fs::read(restored).unwrap(),
        std::fs::read(base).unwrap()
    );
}

#[test]
fn mixed_precision_layout_can_reorder_tensors_but_not_change_shapes_or_dtypes() {
    fn save(root: &Path, name: &str, header: &str, data: &[u16]) -> (std::path::PathBuf, String) {
        let path = root.join(format!("{name}.safetensors"));
        let mut fixture = (header.len() as u64).to_le_bytes().to_vec();
        fixture.extend_from_slice(header.as_bytes());
        fixture.extend(data.iter().flat_map(|v| v.to_le_bytes()));
        std::fs::write(&path, &fixture).unwrap();
        (path, format!("{:x}", Sha256::digest(fixture)))
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let base_header = r#"{"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"h":{"dtype":"F16","shape":[1],"data_offsets":[2,4]}}"#;
    let peer_header = r#"{"h":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#;
    let (base, pin) = save(&root, "base", base_header, &[0x3f80, 0x3c00]);
    let (a, ap) = save(&root, "a", peer_header, &[0x4000, 0x4000]);
    for (i, bad) in [
        peer_header.replace("\"F16\"", "\"BF16\""),
        peer_header.replace("[1]", "[]"),
        peer_header.replace("\"h\"", "\"x\""),
    ]
    .into_iter()
    .enumerate()
    {
        let (path, hash) = save(&root, &format!("bad-{i}"), &bad, &[0x4000, 0x4000]);
        let inputs = [Input {
            path: &path,
            sha256: &hash,
            weight: 0.5,
        }];
        let error = build(&base, &pin, &root, Method::TaskArithmetic, &inputs)
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("names, shapes or dtypes"),
            "case {i}: {error:#}"
        );
        assert!(!root.join(MODEL).exists());
    }
    let inputs = [Input {
        path: &a,
        sha256: &ap,
        weight: 0.5,
    }];
    let result = build(&base, &pin, &root, Method::TaskArithmetic, &inputs).unwrap();
    let (_, expected) = save(&root, "expected", base_header, &[0x3fc0, 0x3e00]);
    assert_eq!(result.sha256, expected);
    assert_eq!(result.delta.changed_tensors, 2);
}
