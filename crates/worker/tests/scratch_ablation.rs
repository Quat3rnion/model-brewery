use brewery_worker::weight_source::scratch::{
    self,
    ablation::{Kernel, Plan, Settings, Shape, Variant},
};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use std::fs;

const WRITERS: [&str; 2] = ["self_attn.o_proj", "mlp.down_proj"];

#[test]
fn ablates_residual_writers_and_reverts_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let path = root.join("source.safetensors");
    // W is [output, input]: its columns write to the residual stream.
    let values = [3_f32, 1., 4., 4., 0., 3., 2., 2., 2.];
    let mut header = Map::new();
    let mut payload = Vec::new();
    for layer in 0..3 {
        for name in WRITERS {
            let start = payload.len();
            payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(
                format!("model.layers.{layer}.{name}.weight"),
                json!({"dtype": "F32", "shape": [3,3], "data_offsets": [start,payload.len()]}),
            );
        }
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut original = (header.len() as u64).to_le_bytes().to_vec();
    original.extend(&header);
    original.extend(payload);
    fs::write(&path, &original).unwrap();
    let sha = format!("{:x}", Sha256::digest(&original));
    let close = |got: f32, want: f32| assert!((got - want).abs() < 1e-5, "{got} != {want}");
    for (label, variant, direction) in [
        ("plain", Variant::Plain, [1., 0., 0.]),
        ("projected", Variant::Projected, [1., 1., 0.]),
        ("norm", Variant::NormPreserving, [1., 1., 1.]),
    ] {
        let dest = root.join(label);
        fs::create_dir(&dest).unwrap();
        let settings = Settings {
            first_layer: 0,
            last_layer: 1,
            kernel: Kernel {
                shape: Shape::Triangle,
                peak: 0,
                width: 1.,
                min_weight: 0.5,
            },
            variant,
        };
        let plan = Plan::new(&settings, &direction, Some(&[0., 1., 0.]), |_| {
            Some(&[0., 1., 1.])
        })
        .unwrap();
        let result = scratch::build_ablated(&path, &sha, &dest, &plan).unwrap();
        assert_eq!(result.delta.changed_tensors, 4);
        let child = fs::read(dest.join(scratch::MODEL)).unwrap();
        let edited: Vec<_> = child[8 + header.len()..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(edited[36..], values.repeat(2));
        for (i, tensor) in edited[..36].chunks_exact(9).enumerate() {
            for col in 0..3 {
                if variant != Variant::NormPreserving || i < 2 {
                    close(tensor[col], if i < 2 { 0. } else { values[col] * 0.5 });
                }
                if variant == Variant::Projected {
                    close(tensor[3 + col], values[3 + col]);
                    close(tensor[6 + col], values[6 + col]);
                }
                if variant == Variant::NormPreserving {
                    let norm_sq = |x: &[f32]| (0..3).map(|r| x[r * 3 + col].powi(2)).sum();
                    close(norm_sq(tensor), norm_sq(&values));
                }
            }
        }
        let restored = dest.join("restored.safetensors");
        scratch::revert(
            &dest.join(scratch::MODEL),
            &dest.join(scratch::DELTA),
            &restored,
        )
        .unwrap();
        assert_eq!(fs::read(restored).unwrap(), original);
    }
    assert_eq!(fs::read(path).unwrap(), original);
}
