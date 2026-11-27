//! Inspect checkpoint dimensions and hash on the same opened Library file.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs::File, io::Read, os::unix::fs::MetadataExt, path::PathBuf};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub path: PathBuf,
    pub kind: String,
    pub sha256: String,
}

impl Request {
    pub fn check(&self) -> Result<()> {
        crate::absolute(&self.path)?;
        ensure!(crate::is_digest(&self.sha256), "invalid model SHA-256");
        let extension = if crate::trace::is_hgn(&self.kind) {
            "hgn"
        } else {
            ensure!(
                matches!(self.kind.as_str(), "gguf" | "safetensors"),
                "unsupported direction model format"
            );
            self.kind.as_str()
        };
        ensure!(
            self.path.extension().is_some_and(|part| part == extension),
            "expected a standalone checkpoint"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dimensions {
    pub hidden_size: usize,
    pub layer_count: usize,
}

pub fn inspect(mut file: File, request: &Request) -> Result<Dimensions> {
    request.check()?;
    let before = file.metadata()?;
    ensure!(
        before.is_file() && (10..=256 << 30).contains(&before.len()),
        "invalid checkpoint file size"
    );
    let dimensions = match request.kind.as_str() {
        "safetensors" => {
            let header = crate::weight_source::Header::read(&mut file, before.len(), 16 << 20)?;
            dimensions(
                header
                    .entries
                    .iter()
                    .map(|(name, t)| (name.as_str(), t.shape.as_slice())),
                "model.embed_tokens.weight",
                &["model.layers."],
            )?
        }
        "gguf" => crate::weight_spectra::gguf_dimensions(&mut file, before.len())?,
        _ => hgn_dimensions(&mut file, before.len())?,
    };
    ensure!(
        crate::weight_source::scratch::file_sha256(&mut file)? == request.sha256,
        "checkpoint SHA-256 mismatch"
    );
    let after = file.metadata()?;
    ensure!(
        before.len() == after.len()
            && before.modified()? == after.modified()?
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec(),
        "checkpoint changed during inspection"
    );
    Ok(dimensions)
}

pub(crate) fn dimensions<'a>(
    tensors: impl Iterator<Item = (&'a str, &'a [u64])>,
    embedding: &str,
    layer_prefixes: &[&str],
) -> Result<Dimensions> {
    let mut width = None;
    let mut layers = BTreeSet::new();
    for (name, shape) in tensors {
        if name == embedding
            && let [vocab, hidden] = *shape
        {
            ensure!(
                width.is_none() && vocab != 0 && (1..=16384).contains(&hidden),
                "invalid embedding dimensions"
            );
            width = Some(hidden as usize);
        }
        if let Some(layer) = layer_prefixes
            .iter()
            .find_map(|prefix| name.strip_prefix(prefix))
            .and_then(|tail| tail.split_once('.'))
            .and_then(|(index, _)| index.parse::<usize>().ok())
        {
            ensure!(layer < 4096, "checkpoint has too many layers");
            layers.insert(layer);
        }
    }
    ensure!(
        !layers.is_empty() && layers.iter().copied().eq(0..layers.len()),
        "expected contiguous transformer layers"
    );
    Ok(Dimensions {
        hidden_size: width.context("missing token embedding dimensions")?,
        layer_count: layers.len(),
    })
}

// Read just the canonical HGN v2 table: the container Reader takes a path,
// but inspection must hash the exact same file descriptor, not reopen a path.
fn hgn_dimensions(file: &mut File, size: u64) -> Result<Dimensions> {
    use bf16_to_hgn::container::{ENTRY_SIZE, HEADER_SIZE, align};
    let mut header = [0; HEADER_SIZE as usize];
    file.read_exact(&mut header)?;
    let u64_at =
        |bytes: &[u8], offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    let count = u64_at(&header, 8);
    ensure!(
        &header[..8] == b"HGN1\x02\x00\x00\x00" && (1..=65536).contains(&count),
        "expected a bounded HGN v2 table"
    );
    let data = align(HEADER_SIZE + ENTRY_SIZE * count, 64)?;
    ensure!(
        u64_at(&header, 16) == HEADER_SIZE
            && u64_at(&header, 24) == data
            && u64_at(&header, 32) == size
            && size >= data,
        "invalid HGN table extent"
    );
    let mut entries = Vec::with_capacity(count as usize);
    let mut names = BTreeSet::new();
    for _ in 0..count {
        let mut entry = [0; ENTRY_SIZE as usize];
        file.read_exact(&mut entry)?;
        let end = entry[..96].iter().position(|&b| b == 0).unwrap_or(96);
        let name = std::str::from_utf8(&entry[..end])?.to_owned();
        ensure!(
            !name.is_empty() && names.insert(name.clone()),
            "invalid HGN tensor name"
        );
        let rank = u32::from_le_bytes(entry[100..104].try_into().unwrap()) as usize;
        ensure!((1..=4).contains(&rank), "invalid HGN tensor rank");
        entries.push((
            name,
            (0..rank)
                .map(|axis| u64_at(&entry, 104 + axis * 8))
                .collect::<Vec<_>>(),
        ));
    }
    dimensions(
        entries
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        "embed_tokens.weight",
        &["layers."],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn pin_and_inspect(kind: &str, bytes: &[u8]) -> Dimensions {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("tiny.{kind}"));
        let mut request = Request {
            path: path.clone(),
            kind: kind.into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        };
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        drop(file);
        let result = inspect(File::open(&path).unwrap(), &request).unwrap();
        assert_eq!((result.hidden_size, result.layer_count), (3, 2));
        request.sha256 = "0".repeat(64);
        assert!(inspect(File::open(path).unwrap(), &request).is_err());
        result
    }

    #[test]
    fn inspects_pinned_safetensors_dimensions() {
        let header = br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"model.layers.0.input_layernorm.weight":{"dtype":"F32","shape":[3],"data_offsets":[24,36]},"model.layers.1.input_layernorm.weight":{"dtype":"F32","shape":[3],"data_offsets":[36,48]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0; 48]);
        pin_and_inspect("safetensors", &bytes);
    }

    #[test]
    fn inspects_pinned_gguf_dimensions() {
        let mut bytes = b"GGUF\x03\x00\x00\x00".to_vec();
        bytes.extend(3u64.to_le_bytes()); // tensor count
        bytes.extend(0u64.to_le_bytes()); // metadata count
        for (i, (name, shape)) in [
            ("token_embd.weight", &[3u64, 2][..]),
            ("blk.0.attn_norm.weight", &[3][..]),
            ("blk.1.attn_norm.weight", &[3][..]),
        ]
        .into_iter()
        .enumerate()
        {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend((shape.len() as u32).to_le_bytes());
            bytes.extend(shape.iter().flat_map(|axis| axis.to_le_bytes()));
            bytes.extend(0u32.to_le_bytes()); // F32
            bytes.extend((i as u64 * 32).to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32) + 3 * 32, 0);
        pin_and_inspect("gguf", &bytes);
    }
}
