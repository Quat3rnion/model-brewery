use super::*;
use crate::weight_source::{Header, open_regular, scratch::decode};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, Metadata},
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::PathBuf,
};

mod gguf;

const MAX_FILES: usize = 128;
const MAX_TENSORS: usize = 8192;
const MAX_TOTAL_BYTES: u64 = 256 << 30;
const HEADER_BYTES: u64 = 4 << 20;

#[derive(Serialize)]
pub(super) struct Receipt {
    id: String,
    sha256: String,
    format: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    shards: Vec<(String, String)>,
}

pub(super) struct Checkpoint {
    pub entries: BTreeMap<String, Entry>,
    files: Vec<(File, Metadata)>,
    pub receipt: Receipt,
}

pub(super) struct Entry {
    pub name: String,
    shape: Vec<u64>,
    dtype: String,
    offset: u64,
    file: usize,
}

impl Checkpoint {
    pub fn open(
        (id, path, kind, expected): Input<'_>,
        cancelled: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let paths = if kind == "hf" {
            let mut paths = Vec::new();
            for (n, child) in fs::read_dir(path)?.enumerate() {
                ensure!(n < 4096, "Too many checkpoint directory entries");
                let path = child?.path();
                if path.extension().is_some_and(|ext| ext == "safetensors") {
                    paths.push(path);
                    ensure!(paths.len() <= MAX_FILES, "Too many SafeTensors shards");
                }
            }
            paths.sort_unstable();
            ensure!(
                !paths.is_empty() && expected.is_none(),
                "Expected a SafeTensors directory"
            );
            paths
        } else {
            ensure!(
                matches!(kind, "safetensors" | "gguf"),
                "Unsupported checkpoint format"
            );
            vec![path.to_owned()]
        };
        let mut entries = BTreeMap::new();
        let mut files = Vec::new();
        let mut hashes = Vec::new();
        let mut buffer = vec![0; 1 << 20];
        let mut size = 0u64;
        for path in &paths {
            cancelled()?;
            let mut fd = open_regular(path).context("Cannot open a checkpoint file")?;
            let metadata = fd.metadata()?;
            size = size
                .checked_add(metadata.len())
                .context("Checkpoint size overflow")?;
            ensure!(
                size <= MAX_TOTAL_BYTES,
                "Checkpoint exceeds the spectral job byte limit"
            );
            let shard = files.len();
            if kind == "gguf" {
                entries = gguf::inventory(&mut fd, metadata.len())?;
            } else {
                let h = Header::read(&mut fd, metadata.len(), HEADER_BYTES)?;
                for (name, t) in h.entries {
                    insert(
                        &mut entries,
                        name.clone(),
                        Entry {
                            name,
                            shape: t.shape,
                            dtype: t.dtype,
                            offset: h.payload_start + t.data_offsets[0],
                            file: shard,
                        },
                    )?;
                }
            }
            fd.rewind()?;
            let mut sha = Sha256::new();
            let mut remaining = metadata.len();
            while remaining > 0 {
                cancelled()?;
                let count = remaining.min(buffer.len() as u64) as usize;
                fd.read_exact(&mut buffer[..count])?;
                sha.update(&buffer[..count]);
                remaining -= count as u64;
            }
            let digest = format!("{:x}", sha.finalize());
            ensure!(
                expected.is_none_or(|pin| pin == digest),
                "Checkpoint hash does not match its pinned revision"
            );
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .context("Invalid shard filename")?;
            hashes.push((name.to_string(), digest));
            files.push((fd, metadata));
        }
        if paths.len() > 1 {
            verify_index(path, &entries, &paths)?;
        }
        let sha256 = if hashes.len() == 1 {
            hashes[0].1.clone()
        } else {
            // Bind shard names and hashes, independently of filesystem order.
            format!("{:x}", Sha256::digest(serde_json::to_vec(&hashes)?))
        };
        let receipt = Receipt {
            id: id.into(),
            sha256,
            format: if kind == "gguf" { kind } else { "safetensors" }.into(),
            shards: if hashes.len() > 1 { hashes } else { Vec::new() },
        };
        let result = Self {
            entries,
            files,
            receipt,
        };
        result.verify_unchanged()?;
        Ok(result)
    }

    pub fn verify_unchanged(&self) -> Result<()> {
        for (file, old) in &self.files {
            let now = file.metadata()?;
            ensure!(
                (
                    now.len(),
                    now.ctime(),
                    now.ctime_nsec(),
                    now.mtime(),
                    now.mtime_nsec()
                ) == (
                    old.len(),
                    old.ctime(),
                    old.ctime_nsec(),
                    old.mtime(),
                    old.mtime_nsec()
                ),
                "Checkpoint changed during the spectral job"
            );
        }
        Ok(())
    }

    pub fn measure(
        &mut self,
        key: &str,
        cancelled: &mut impl FnMut() -> Result<()>,
    ) -> Result<Option<Measurement>> {
        let Some(t) = self.entries.get(key) else {
            return Ok(None);
        };
        let mut result = Measurement {
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            metrics: None,
            reason: None,
        };
        let layout = quant_layout(&t.dtype);
        let (rows, cols, block, bytes) = match (t.shape.as_slice(), layout) {
            ([r, c], Some((block, bytes))) => (*r, *c, block, bytes),
            ([c], Some((block, bytes))) => (1, *c, block, bytes),
            (_, None) => {
                result.reason = Some("unsupported_dtype");
                return Ok(Some(result));
            }
            _ => {
                result.reason = Some("not_a_matrix_or_vector");
                return Ok(Some(result));
            }
        };
        if !(1..=math::MAX_AXIS as u64).contains(&rows)
            || !(1..=math::MAX_AXIS as u64).contains(&cols)
            || rows * cols > (1 << 30)
        {
            result.reason = Some("shape_outside_compute_limits");
            return Ok(Some(result));
        }
        let (rows, cols) = (rows as usize, cols as usize);
        ensure!(cols % block == 0, "Invalid quantized row width");
        let row_bytes = cols / block * bytes;
        let fd = &mut self.files[t.file].0;
        let mut raw = Vec::new();
        let mut values = Vec::new();
        let metrics = math::measure(rows, cols, |start, count| {
            cancelled()?;
            raw.resize(count * row_bytes, 0);
            fd.seek(SeekFrom::Start(t.offset + (start * row_bytes) as u64))?;
            fd.read_exact(&mut raw)?;
            values.clear();
            for chunk in raw.chunks_exact(bytes) {
                if block == 1 {
                    values.push(decode(&t.dtype, chunk));
                } else {
                    gguf::dequantize(&t.dtype, chunk, &mut values);
                }
            }
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "Non-finite weight in {key}"
            );
            Ok(DMatrix::from_row_slice(count, cols, &values))
        })?;
        result.metrics = Some(metrics);
        Ok(Some(result))
    }
}

fn verify_index(root: &Path, entries: &BTreeMap<String, Entry>, paths: &[PathBuf]) -> Result<()> {
    let mut file = open_regular(&root.join(crate::weight_source::INDEX))
        .context("Sharded SafeTensors needs an index")?;
    let limit = file.metadata()?.len();
    ensure!(limit <= HEADER_BYTES, "SafeTensors index too large");
    let mut data = vec![0; limit as usize];
    file.read_exact(&mut data)?;
    let index: Value = serde_json::from_slice(&data)?;
    let map = index["weight_map"]
        .as_object()
        .context("Missing SafeTensors weight_map")?;
    ensure!(
        map.len() == entries.len()
            && entries.iter().all(|(name, t)| {
                map.get(name).and_then(Value::as_str)
                    == paths[t.file].file_name().and_then(|s| s.to_str())
            }),
        "SafeTensors shard index disagrees with tensor files"
    );
    Ok(())
}

fn insert(entries: &mut BTreeMap<String, Entry>, key: String, value: Entry) -> Result<()> {
    ensure!(entries.len() < MAX_TENSORS, "Too many checkpoint tensors");
    ensure!(
        entries.insert(key, value).is_none(),
        "Duplicate checkpoint tensor"
    );
    Ok(())
}

fn quant_layout(dtype: &str) -> Option<(usize, usize)> {
    Some(match dtype {
        "F32" => (1, 4),
        "F16" | "BF16" => (1, 2),
        "Q4_0" => (32, 18),
        "Q8_0" => (32, 34),
        "Q4_K" => (256, 144),
        "Q6_K" => (256, 210),
        _ => return None,
    })
}
