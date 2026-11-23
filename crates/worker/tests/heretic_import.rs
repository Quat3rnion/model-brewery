use brewery_worker::weight_source::scratch::{DELTA, MODEL, import, revert};
use sha2::{Digest, Sha256};
use std::fs;

fn model(bits: u32, extra: &str) -> Vec<u8> {
    let header =
        format!(r#"{{"weight":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}{extra}}}"#);
    [
        (header.len() as u64).to_le_bytes().as_slice(),
        header.as_bytes(),
        &bits.to_le_bytes(),
    ]
    .concat()
}

#[test]
fn imports_exact_tensor_bytes_and_restores_the_library_parent() {
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("parent.safetensors");
    let exported = dir.path().join("exported.safetensors");
    let child = dir.path().join("child");
    fs::create_dir(&child).unwrap();
    let original = model(0, "");
    let edited = model(0x8000_0000, r#","__metadata__":{"format":"pt"}"#);
    fs::write(&parent, &original).unwrap();
    fs::write(&exported, &edited).unwrap();
    let parent_hash = format!("{:x}", Sha256::digest(&original));
    let exported_hash = format!("{:x}", Sha256::digest(&edited));
    assert!(import(&parent, &parent_hash, &exported, &"0".repeat(64), &child).is_err());
    let result = import(&parent, &parent_hash, &exported, &exported_hash, &child).unwrap();
    assert_eq!(result.delta.changed_tensors, 1);
    assert_eq!(fs::read(child.join(MODEL)).unwrap(), model(0x8000_0000, ""));
    let restored = dir.path().join("restored");
    revert(&child.join(MODEL), &child.join(DELTA), &restored).unwrap();
    assert_eq!(fs::read(restored).unwrap(), original);
}
