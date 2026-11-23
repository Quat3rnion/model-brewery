//! Byte-preserving direct edits for single-file SafeTensors checkpoints.
use super::{Header, Tensor, native_tensor_name, open_regular};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

pub const MODEL: &str = "model.safetensors";
pub const DELTA: &str = "delta.bin";
pub const MAX_FILE: u64 = 4 << 30;
pub const MAX_EDIT_BYTES: u64 = 32 << 20;
const DELTA_HEADER: u64 = 8 + 64 + 64 + 8 + 4;
pub const DELTA_OVERHEAD: u64 = DELTA_HEADER + 32 * 16;
const MAX_DELTA_RECORDS: usize = 4096;
pub const MERGE_DELTA_OVERHEAD: u64 = DELTA_HEADER + MAX_DELTA_RECORDS as u64 * 16;
const MAGIC: &[u8; 8] = b"BWDELTA1";

pub mod diff;
pub mod merge;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Set {
        tensor: String,
        origin: Vec<u64>,
        shape: Vec<u64>,
        values: Vec<f64>,
    },
    Add {
        tensor: String,
        origin: Vec<u64>,
        shape: Vec<u64>,
        values: Vec<f64>,
    },
    LowRank {
        tensor: String,
        u: Vec<f64>,
        v: Vec<f64>,
        scale: f64,
    },
    Orthogonalize {
        tensor: String,
        axis: Axis,
        direction: Vec<f64>,
    },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Rows,
    Columns,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Summary {
    pub schema: String,
    pub parent_sha256: String,
    pub revision_sha256: String,
    pub changed_tensors: usize,
    pub changed_bytes: u64,
    pub delta_bytes: u64,
    pub delta_sha256: String,
}

pub struct Output {
    pub sha256: String,
    pub tensors: Vec<TensorInfo>,
    pub delta: Summary,
}

impl Operation {
    fn name(&self) -> &str {
        match self {
            Self::Set { tensor, .. }
            | Self::Add { tensor, .. }
            | Self::LowRank { tensor, .. }
            | Self::Orthogonalize { tensor, .. } => tensor,
        }
    }

    fn apply(&self, t: &Tensor, bytes: &mut [u8]) -> Result<()> {
        let dtype = t.dtype.as_str();
        let width = editable_width(dtype)?;
        let dims = &t.shape;
        let mut update = |index: usize, f: &dyn Fn(f64) -> f64| -> Result<()> {
            let value = &mut bytes[index * width..(index + 1) * width];
            let old = decode(dtype, value);
            ensure!(old.is_finite(), "non-finite input weight");
            encode(dtype, f(old), value)
        };
        match self {
            Self::Set {
                origin,
                shape,
                values,
                ..
            }
            | Self::Add {
                origin,
                shape,
                values,
                ..
            } => {
                ensure!(
                    dims.len() == shape.len()
                        && dims.len() == origin.len()
                        && (1..=8).contains(&dims.len())
                        && origin.iter().zip(shape).zip(dims).all(|((&o, &s), &d)| {
                            s > 0 && o.checked_add(s).is_some_and(|end| end <= d)
                        }),
                    "invalid slice"
                );
                let count: u64 = shape.iter().product();
                ensure!(
                    count == values.len() as u64,
                    "slice value count does not match shape"
                );
                let add = matches!(self, Self::Add { .. });
                for (mut position, &operand) in values.iter().enumerate() {
                    let mut index = 0;
                    let mut stride = 1;
                    for ((&o, &s), &d) in origin.iter().zip(shape).zip(dims).rev() {
                        index += (o as usize + position % s as usize) * stride;
                        position /= s as usize;
                        stride *= d as usize;
                    }
                    update(index, &|old| if add { old + operand } else { operand })?;
                }
            }
            Self::LowRank { u, v, scale, .. } => {
                ensure!(
                    dims.as_slice() == [u.len() as u64, v.len() as u64],
                    "low_rank requires a matrix with shape [len(u), len(v)]"
                );
                for (r, &left) in u.iter().enumerate() {
                    for (c, &right) in v.iter().enumerate() {
                        update(r * v.len() + c, &|old| old + scale * left * right)?;
                    }
                }
            }
            Self::Orthogonalize {
                axis, direction, ..
            } => {
                let [rows, cols] = dims.as_slice() else {
                    bail!("orthogonalize requires a matrix")
                };
                let (rows, cols) = (*rows as usize, *cols as usize);
                let (vectors, length, stride, step) = match axis {
                    Axis::Rows => (rows, cols, cols, 1),
                    Axis::Columns => (cols, rows, 1, cols),
                };
                ensure!(
                    direction.len() == length,
                    "direction length does not match axis"
                );
                let max = direction.iter().map(|v| v.abs()).fold(0.0, f64::max);
                ensure!(
                    max > 0.0 && max.is_finite(),
                    "direction must be finite and nonzero"
                );
                let direction: Vec<_> = direction.iter().map(|v| v / max).collect();
                let norm_sq: f64 = direction.iter().map(|v| v * v).sum();
                for i in 0..vectors {
                    let start = i * stride * width;
                    let inner: f64 = direction
                        .iter()
                        .enumerate()
                        .map(|(j, d)| {
                            let offset = start + j * step * width;
                            d * decode(dtype, &bytes[offset..offset + width])
                        })
                        .sum();
                    ensure!(inner.is_finite(), "non-finite input weight");
                    for (j, d) in direction.iter().enumerate() {
                        let offset = start + j * step * width;
                        let value = &mut bytes[offset..offset + width];
                        encode(dtype, decode(dtype, value) - inner / norm_sq * d, value)?;
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn build(
    source: &Path,
    expected: &str,
    dir: &Path,
    operations: &[Operation],
) -> Result<Output> {
    ensure!(crate::is_digest(expected), "expected a lowercase SHA-256");
    ensure!(
        operations.len() <= 32,
        "at most 32 edit operations are allowed"
    );
    ensure!(
        source.extension().is_some_and(|ext| ext == "safetensors"),
        "expected a .safetensors source"
    );
    let source = open_regular(source)?;
    let size = source.metadata()?.len();
    ensure!(
        (10..=MAX_FILE).contains(&size),
        "SafeTensors file exceeds scratch limit"
    );
    let mut child = new_file(&dir.join(MODEL))?;
    let count = io::copy(&mut source.take(size + 1), &mut child)?;
    ensure!(count == size, "source size changed");
    ensure!(
        file_sha256(&mut child)? == expected,
        "parent SHA-256 mismatch"
    );
    let header = Header::read(&mut child, size, super::LIMIT)?;

    // Group by tensor but preserve the order of operations on each tensor.
    let mut groups = BTreeMap::<_, Vec<_>>::new();
    for op in operations {
        ensure!(native_tensor_name(op.name()), "invalid tensor name");
        groups.entry(op.name()).or_default().push(op);
    }
    let mut tensors = Vec::new();
    let mut records = Vec::new();
    let mut edited = 0;
    let mut changed_bytes = 0;
    for (name, ops) in groups {
        let t = header.entries.get(name).context("unknown tensor")?;
        let [start, end] = t.data_offsets;
        edited += end - start;
        ensure!(
            end > start && edited <= MAX_EDIT_BYTES,
            "edited tensors exceed scratch limit or are empty"
        );
        let offset = header.payload_start + start;
        let mut before = vec![0; (end - start) as usize];
        child.seek(SeekFrom::Start(offset))?;
        child.read_exact(&mut before)?;
        let mut after = before.clone();
        for op in ops {
            op.apply(t, &mut after)?;
        }
        for (a, b) in before.iter_mut().zip(&after) {
            *a ^= *b;
            changed_bytes += u64::from(*a != 0);
        }
        if let Some(lo) = before.iter().position(|b| *b != 0) {
            let hi = before.iter().rposition(|b| *b != 0).unwrap() + 1;
            let offset = offset + lo as u64;
            child.seek(SeekFrom::Start(offset))?;
            child.write_all(&after[lo..hi])?;
            before.truncate(hi);
            before.drain(..lo);
            records.push((offset, before));
        }
        tensors.push(TensorInfo {
            name: name.into(),
            shape: t.shape.clone(),
            dtype: t.dtype.clone(),
        });
    }
    ensure!(
        operations.is_empty() || !records.is_empty(),
        "no stored weight changes"
    );
    child.sync_all()?;
    let sha256 = if operations.is_empty() {
        expected.to_owned()
    } else {
        file_sha256(&mut child)?
    };
    records.sort_unstable_by_key(|(offset, _)| *offset);
    let mut delta = new_file(&dir.join(DELTA))?;
    delta.write_all(MAGIC)?;
    delta.write_all(expected.as_bytes())?;
    delta.write_all(sha256.as_bytes())?;
    delta.write_all(&size.to_le_bytes())?;
    delta.write_all(&(records.len() as u32).to_le_bytes())?;
    for (offset, xor) in &records {
        delta.write_all(&offset.to_le_bytes())?;
        delta.write_all(&(xor.len() as u64).to_le_bytes())?;
        delta.write_all(xor)?;
    }
    delta.sync_all()?;
    Ok(Output {
        sha256: sha256.clone(),
        tensors,
        delta: Summary {
            schema: "brewery.weight-delta.v1".into(),
            parent_sha256: expected.into(),
            revision_sha256: sha256,
            changed_tensors: records.len(),
            changed_bytes,
            delta_bytes: delta.metadata()?.len(),
            delta_sha256: file_sha256(&mut delta)?,
        },
    })
}

/// Reconstruct the pinned parent in a new file. Do not write to the child
/// or to an existing destination, even if a delta is invalid.
pub fn revert(child: &Path, delta: &Path, destination: &Path) -> Result<()> {
    fn read<const N: usize>(file: &mut File) -> Result<[u8; N]> {
        let mut value = [0; N];
        file.read_exact(&mut value)?;
        Ok(value)
    }
    let mut delta = open_regular(delta)?;
    ensure!(
        delta.metadata()?.len() <= MAX_FILE + MERGE_DELTA_OVERHEAD,
        "scratch delta exceeds limit"
    );
    ensure!(
        &read::<8>(&mut delta)? == MAGIC,
        "invalid scratch delta magic"
    );
    let parent_hash = read::<64>(&mut delta)?;
    let child_hash = read::<64>(&mut delta)?;
    let size = u64::from_le_bytes(read(&mut delta)?);
    let records = u32::from_le_bytes(read(&mut delta)?);
    ensure!(
        (10..=MAX_FILE).contains(&size) && records as usize <= MAX_DELTA_RECORDS,
        "invalid scratch delta size or record count"
    );
    let child = open_regular(child)?;
    ensure!(
        child.metadata()?.len() == size,
        "scratch child size mismatch"
    );
    let dir = destination
        .parent()
        .context("restore destination needs a parent directory")?;
    let mut restored = tempfile::NamedTempFile::new_in(dir)?;
    let output = restored.as_file_mut();
    ensure!(
        io::copy(&mut child.take(size + 1), output)? == size,
        "scratch child size changed"
    );
    ensure!(
        file_sha256(output)?.as_bytes() == child_hash,
        "scratch delta belongs to a different child hash"
    );
    let mut last_end = 0;
    let mut a = vec![0; 64 << 10];
    let mut b = a.clone();
    for _ in 0..records {
        let offset = u64::from_le_bytes(read(&mut delta)?);
        let length = u64::from_le_bytes(read(&mut delta)?);
        ensure!(
            length > 0
                && offset >= last_end
                && offset.checked_add(length).is_some_and(|end| end <= size),
            "invalid scratch delta range"
        );
        last_end = offset + length;
        output.seek(SeekFrom::Start(offset))?;
        let mut left = length;
        while left != 0 {
            let count = left.min(a.len() as u64) as usize;
            output.read_exact(&mut a[..count])?;
            delta.read_exact(&mut b[..count])?;
            for i in 0..count {
                a[i] ^= b[i];
            }
            output.seek(SeekFrom::Current(-(count as i64)))?;
            output.write_all(&a[..count])?;
            left -= count as u64;
        }
    }
    ensure!(
        delta.read(&mut a[..1])? == 0,
        "unexpected bytes after scratch delta records"
    );
    ensure!(
        file_sha256(output)?.as_bytes() == parent_hash,
        "restored scratch parent hash mismatch"
    );
    output.sync_all()?;
    restored.persist_noclobber(destination)?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn editable_width(dtype: &str) -> Result<usize> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        _ => bail!("scratch edits require F32, F16, or BF16"),
    }
}

fn decode(dtype: &str, bytes: &[u8]) -> f64 {
    if dtype == "F32" {
        return f32::from_le_bytes(bytes.try_into().unwrap()).into();
    }
    let bits = u16::from_le_bytes(bytes.try_into().unwrap());
    if dtype == "BF16" {
        return f32::from_bits(u32::from(bits) << 16).into();
    }
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let magnitude = match exponent {
        0 => f64::from(fraction) * 2f64.powi(-24),
        31 => return f64::NAN,
        _ => f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25),
    };
    if bits >> 15 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn encode(dtype: &str, value: f64, dst: &mut [u8]) -> Result<()> {
    ensure!(value.is_finite(), "non-finite edited weight");
    if dtype == "F32" {
        let value = value as f32;
        ensure!(value.is_finite(), "edited weight exceeds finite F32 range");
        dst.copy_from_slice(&value.to_le_bytes());
        return Ok(());
    }
    let (fraction, bias) = if dtype == "F16" { (10, 15) } else { (7, 127) };
    let absolute = value.abs();
    // Round directly from F64, not through F32 (double rounding at ties).
    let exponent = ((absolute.to_bits() >> 52) as i32 - 1023).max(1 - bias);
    ensure!(
        exponent <= bias,
        "edited weight exceeds finite {dtype} range"
    );
    let rounded = (absolute * 2f64.powi(fraction - exponent)).round_ties_even() as u16;
    let bits = (((exponent + bias - 1) as u16) << fraction) + rounded;
    ensure!(
        bits < ((2 * bias + 1) as u16) << fraction,
        "edited weight exceeds finite {dtype} range"
    );
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    dst.copy_from_slice(&(bits | sign).to_le_bytes());
    Ok(())
}

fn new_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .mode(0o400)
        .open(path)
}

pub fn file_sha256(file: &mut File) -> Result<String> {
    file.rewind()?;
    let mut sha = Sha256::new();
    io::copy(file, &mut sha)?;
    file.rewind()?;
    Ok(format!("{:x}", sha.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_half_writes_round_directly_to_the_storage_type() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let path = root.join("source.safetensors");
        let header = br#"{"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"h":{"dtype":"F16","shape":[2],"data_offsets":[4,8]}}"#;
        let mut fixture = (header.len() as u64).to_le_bytes().to_vec();
        fixture.extend(header);
        fixture.extend([0; 8]);
        std::fs::write(&path, &fixture).unwrap();
        let operations = [("b", 8), ("h", 11)].map(|(name, shift)| {
            let tie = 1. + 2f64.powi(-shift);
            Operation::Set {
                tensor: name.into(),
                origin: vec![0],
                shape: vec![2],
                values: vec![tie, tie + 2f64.powi(-40)],
            }
        });
        let pin = format!("{:x}", Sha256::digest(&fixture));
        let outcome = build(&path, &pin, &root, &operations).unwrap();
        let result = std::fs::read(root.join(MODEL)).unwrap();
        assert_eq!(
            &result[result.len() - 8..],
            &[0x80, 0x3f, 0x81, 0x3f, 0, 0x3c, 1, 0x3c]
        );
        assert_eq!(outcome.delta.changed_tensors, 2);
        assert_eq!(outcome.delta.changed_bytes, 7);
        assert_eq!(std::fs::read(path).unwrap(), fixture);
    }
}
