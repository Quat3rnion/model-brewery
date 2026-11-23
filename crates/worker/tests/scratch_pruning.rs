use brewery_worker::weight_source::scratch::{self, prune::Plan};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use std::fs;

#[test]
fn structured_pruning_keeps_dimensions_and_masks_whole_groups() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let path = root.join("source.safetensors");
    let tensors: [(&str, &[u64], &[f32]); 5] = [
        (
            "self_attn.o_proj.weight",
            &[4, 4],
            &[
                1., 0., 2., 2., 1., 0., 2., 2., 1., 0., 2., 2., 1., 0., 2., 2.,
            ],
        ),
        ("mlp.down_proj.weight", &[2, 2], &[1., 9., 1., 9.]),
        ("post_attention_layernorm.weight", &[2], &[1., 1.]),
        ("mlp.gate_proj.weight", &[2, 2], &[1., 4., 3., 1.]),
        ("mlp.up_proj.weight", &[2, 2], &[1., 1., 1., 1.]),
    ];
    let mut header = Map::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            format!("model.layers.0.{name}"),
            json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut original = (header.len() as u64).to_le_bytes().to_vec();
    original.extend(&header);
    original.extend(payload);
    fs::write(&path, &original).unwrap();
    let digest = format!("{:x}", Sha256::digest(&original));
    let plans = [
        (
            "head",
            Plan::AttentionHeads {
                layer: 0,
                basis_points: 5000,
                heads: 2,
            },
            0,
            vec![
                0., 0., 2., 2., 0., 0., 2., 2., 0., 0., 2., 2., 0., 0., 2., 2.,
            ],
        ),
        (
            "neuron",
            Plan::MlpNeurons {
                layer: 0,
                basis_points: 5000,
            },
            16,
            vec![0., 9., 0., 9.],
        ),
        // Activation on feature 1 selects the other neuron: gate/up saliency
        // is proportional to [4 + 1, 1 + 1] for x = [0, 1].
        (
            "wanda",
            Plan::WandaMlp {
                layer: 0,
                basis_points: 5000,
                rms_eps: 1e-6,
                trace: scratch::prune::Trace {
                    id: "trace".into(),
                    prompt: 0,
                    start: 0,
                    end: 1,
                },
            },
            16,
            vec![1., 0., 1., 0.],
        ),
    ];
    for (label, plan, start, values) in plans {
        let dest = root.join(label);
        fs::create_dir(&dest).unwrap();
        let result =
            scratch::build_pruned(&path, &digest, &dest, &plan, Some(&[vec![0., 1.]])).unwrap();
        assert_eq!(result.delta.changed_tensors, 1, "{label}");
        let mut expected = original.clone();
        for (i, value) in values.into_iter().enumerate() {
            let offset = 8 + header.len() + (start + i) * 4;
            expected[offset..offset + 4].copy_from_slice(&f32::to_le_bytes(value));
        }
        assert_eq!(
            fs::read(dest.join(scratch::MODEL)).unwrap(),
            expected,
            "{label}"
        );
    }
    assert_eq!(fs::read(path).unwrap(), original);
}
