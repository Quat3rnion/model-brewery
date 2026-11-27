//! Native F32 merges with a fixed-size working block per input.
use super::*;
use std::os::unix::fs::{FileExt, MetadataExt};

const BLOCK: usize = 64 << 10;
pub const MAX_INPUTS: usize = 16;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum Method {
    Linear,
    TaskArithmetic,
    Ties { density: f64 },
    DareLinear { drop_rate: f32, seed: u64 },
    DareTies { drop_rate: f32, seed: u64 },
}

pub struct Input<'a> {
    pub path: &'a Path,
    pub sha256: &'a str,
    pub weight: f32,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::TaskArithmetic => "task_arithmetic",
            Self::Ties { .. } => "ties",
            Self::DareLinear { .. } => "dare_linear",
            Self::DareTies { .. } => "dare_ties",
        }
    }

    fn validate(self, inputs: &[Input<'_>]) -> Result<f32> {
        ensure!(
            (1..=MAX_INPUTS).contains(&inputs.len()),
            "expected 1 to {MAX_INPUTS} merge inputs"
        );
        ensure!(
            inputs.iter().all(|x| x.weight.is_finite()),
            "merge weights must be finite"
        );
        match self {
            Self::Linear => {
                let sum: f32 = inputs.iter().map(|x| x.weight).sum();
                ensure!(
                    inputs.iter().all(|x| x.weight >= 0.) && sum > 0. && sum.is_finite(),
                    "linear weights must be nonnegative with a finite positive sum"
                );
                Ok(sum)
            }
            Self::Ties { density } => {
                ensure!((0. ..=1.).contains(&density), "density must be in [0, 1]");
                Ok(1.)
            }
            Self::DareLinear { drop_rate, .. } | Self::DareTies { drop_rate, .. } => {
                ensure!(
                    (0. ..1.).contains(&drop_rate),
                    "drop_rate must be in [0, 1)"
                );
                Ok(1. / (1. - drop_rate))
            }
            Self::TaskArithmetic => Ok(1.),
        }
    }
}

struct Source {
    file: File,
    header: Header,
    stamp: (u64, i64, i64, i64, i64),
}

fn stamp(file: &File) -> io::Result<(u64, i64, i64, i64, i64)> {
    let m = file.metadata()?;
    Ok((
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}

fn open(path: &Path, hash: &str) -> Result<Source> {
    ensure!(crate::is_digest(hash), "expected a lowercase SHA-256");
    ensure!(
        path.extension().is_some_and(|ext| ext == "safetensors"),
        "expected a .safetensors source"
    );
    let mut file = open_regular(path)?;
    let original = stamp(&file)?;
    let size = original.0;
    ensure!(
        (10..=MAX_FILE).contains(&size),
        "SafeTensors file exceeds scratch limit"
    );
    let header = Header::read(&mut file, size, super::super::LIMIT)?;
    ensure!(
        header.entries.len() <= MAX_DELTA_RECORDS,
        "too many merge tensors"
    );
    file.rewind()?;
    let mut sha = Sha256::new();
    let n = io::copy(&mut (&mut file).take(size + 1), &mut sha)?;
    ensure!(
        n == size && format!("{:x}", sha.finalize()) == hash,
        "merge source SHA-256 mismatch"
    );
    file.rewind()?;
    Ok(Source {
        file,
        header,
        stamp: original,
    })
}

pub fn build(
    base: &Path,
    sha256: &str,
    dir: &Path,
    method: Method,
    inputs: &[Input<'_>],
) -> Result<Output> {
    build_inner(base, sha256, dir, method, inputs, false)
}

/// Import the exact exported tensor bytes in the parent's layout. The W3
/// delta restores the original parent, including its header and padding.
pub fn import(
    base: &Path,
    sha256: &str,
    exported: &Path,
    export_sha256: &str,
    dir: &Path,
) -> Result<Output> {
    build_inner(
        base,
        sha256,
        dir,
        Method::Linear,
        &[Input {
            path: exported,
            sha256: export_sha256,
            weight: 1.,
        }],
        true,
    )
}

fn build_inner(
    base: &Path,
    sha256: &str,
    dir: &Path,
    method: Method,
    inputs: &[Input<'_>],
    import: bool,
) -> Result<Output> {
    let scale = method.validate(inputs)?;
    let mut base = open(base, sha256).context("merge base")?;
    let size = base.stamp.0;
    let mut ordered: Vec<_> = base.header.entries.iter().collect();
    for (_, t) in &ordered {
        editable_width(&t.dtype)?;
    }
    ordered.sort_unstable_by_key(|(_, t)| t.data_offsets[0]);
    let sources = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            open(input.path, input.sha256).with_context(|| format!("merge input {}", index + 1))
        })
        .collect::<Result<Vec<_>>>()?;
    for source in &sources {
        ensure!(
            source.header.entries.len() == ordered.len()
                && ordered.iter().all(|(name, a)| {
                    source
                        .header
                        .entries
                        .get(*name)
                        .is_some_and(|b| a.dtype == b.dtype && a.shape == b.shape)
                }),
            "merge tensor names, shapes or dtypes differ"
        );
    }
    let mut child = new_file(&dir.join(MODEL))?;
    ensure!(
        io::copy(&mut (&mut base.file).take(size + 1), &mut child)? == size,
        "merge base size changed"
    );
    let mut delta = new_file(&dir.join(DELTA))?;
    delta.set_len(DELTA_HEADER)?;
    let mut next = DELTA_HEADER;
    let mut changed_bytes = 0;
    let mut tensors = Vec::new();
    let mut base_buf = vec![0; BLOCK];
    let mut buf = vec![vec![0; BLOCK]; sources.len()];
    let mut after = vec![0; BLOCK];
    let mut xor = vec![0; BLOCK];
    let mut weighted = vec![0.; sources.len()];
    for (name, tensor) in ordered {
        let [start, end] = tensor.data_offsets;
        let size = end - start;
        let offset = base.header.payload_start + start;
        let dtype = tensor.dtype.as_str();
        let width = editable_width(dtype)?;
        let peers: Vec<_> = sources
            .iter()
            .map(|source| {
                (
                    &source.file,
                    source.header.payload_start + source.header.entries[name].data_offsets[0],
                )
            })
            .collect();
        let mut trim = vec![Trim::All; peers.len()];
        if let Method::Ties { density } = method {
            let count = size / width as u64;
            let k = if density == 0. {
                0
            } else {
                ((density * count as f64) as u64).max(1).min(count)
            };
            for (i, &(other, other_offset)) in peers.iter().enumerate() {
                trim[i] = if k == 0 {
                    Trim::None
                } else if k == count {
                    Trim::All
                } else {
                    cutoff(
                        &child,
                        offset,
                        other,
                        other_offset,
                        size,
                        dtype,
                        width,
                        k,
                        &mut base_buf,
                        &mut buf[i],
                    )?
                };
            }
        }
        let mut range = None;
        for pos in (0..size).step_by(BLOCK) {
            let n = (size - pos).min(BLOCK as u64) as usize;
            let old = &mut base_buf[..n];
            child.read_exact_at(old, offset + pos)?;
            for ((file, address), bytes) in peers.iter().zip(&mut buf) {
                file.read_exact_at(&mut bytes[..n], address + pos)?;
            }
            if buf.iter().all(|b| b[..n] == *old) {
                ensure!(
                    old.chunks_exact(width).all(|bytes| finite(dtype, bytes)),
                    "non-finite merge input"
                );
                continue;
            }
            if import {
                ensure!(
                    buf[0][..n]
                        .chunks_exact(width)
                        .all(|bytes| finite(dtype, bytes))
                        && old.chunks_exact(width).all(|bytes| finite(dtype, bytes)),
                    "non-finite import tensor"
                );
                after[..n].copy_from_slice(&buf[0][..n]);
            }
            for i in (0..if import { 0 } else { n }).step_by(width) {
                let bytes = i..i + width;
                let a = decode(dtype, &old[bytes.clone()]) as f32;
                ensure!(a.is_finite(), "non-finite merge base");
                if buf.iter().all(|b| b[bytes.clone()] == old[bytes.clone()]) {
                    after[bytes.clone()].copy_from_slice(&old[bytes]);
                    continue;
                }
                let mut sum = 0f32;
                for (j, input) in inputs.iter().enumerate() {
                    let b = decode(dtype, &buf[j][bytes.clone()]) as f32;
                    ensure!(b.is_finite(), "non-finite merge input");
                    let d = match method {
                        Method::Linear => b,
                        Method::Ties { .. } => trim[j].apply(b - a)?,
                        Method::TaskArithmetic => b - a,
                        Method::DareLinear { drop_rate, seed }
                        | Method::DareTies { drop_rate, seed } => {
                            let index = MAX_FILE * j as u64 + start + pos + i as u64;
                            let delta = b - a;
                            ensure!(delta.is_finite(), "non-finite merge delta");
                            if drop_rate != 0. && uniform(seed, index) < f64::from(drop_rate) {
                                0.
                            } else {
                                delta * scale
                            }
                        }
                    };
                    weighted[j] = d * input.weight;
                    sum += weighted[j];
                }
                ensure!(sum.is_finite(), "non-finite weighted merge sum");
                let value = match method {
                    Method::Linear => sum / scale,
                    Method::Ties { .. } | Method::DareTies { .. } => {
                        let mut chosen = 0.;
                        let mut divisor = 0.;
                        let normalize = matches!(method, Method::Ties { .. });
                        for (&d, input) in weighted.iter().zip(inputs) {
                            if (sum >= 0. && d > 0.) || (sum < 0. && d < 0.) {
                                chosen += d;
                                if normalize {
                                    divisor += input.weight;
                                }
                            }
                        }
                        if normalize && divisor != 0. {
                            ensure!(divisor.is_finite(), "non-finite TIES divisor");
                            chosen /= divisor;
                        }
                        a + chosen
                    }
                    _ => a + sum,
                };
                let out = &mut after[bytes.clone()];
                if value == a {
                    out.copy_from_slice(&old[bytes]);
                } else {
                    encode(dtype, f64::from(value), out)?;
                }
            }
            let xor = &mut xor[..n];
            for ((x, &a), &b) in xor.iter_mut().zip(old.iter()).zip(&after[..n]) {
                *x = a ^ b;
                changed_bytes += u64::from(*x != 0);
            }
            if let Some(lo) = xor.iter().position(|&b| b != 0) {
                let hi = xor.iter().rposition(|&b| b != 0).unwrap() + 1;
                let at = offset + pos;
                child.write_all_at(&after[lo..hi], at + lo as u64)?;
                let (first, last) = range.get_or_insert((at + lo as u64, 0));
                // Seek over unchanged bytes; holes read back as zero XOR.
                delta.write_all_at(&xor[lo..hi], next + 16 + at + lo as u64 - *first)?;
                *last = at + hi as u64;
            }
        }
        if let Some((first, last)) = range {
            delta.write_all_at(&first.to_le_bytes(), next)?;
            delta.write_all_at(&(last - first).to_le_bytes(), next + 8)?;
            next += 16 + last - first;
            tensors.push(TensorInfo {
                name: name.clone(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype.clone(),
            });
        }
    }
    for source in std::iter::once(&base).chain(&sources) {
        ensure!(
            stamp(&source.file)? == source.stamp,
            "merge source changed during the merge"
        );
    }
    child.sync_all()?;
    let revision = file_sha256(&mut child)?;
    delta.write_all(MAGIC)?;
    delta.write_all(sha256.as_bytes())?;
    delta.write_all(revision.as_bytes())?;
    delta.write_all(&size.to_le_bytes())?;
    delta.write_all(&(tensors.len() as u32).to_le_bytes())?;
    delta.sync_all()?;
    Ok(Output {
        sha256: revision.clone(),
        delta: Summary {
            schema: "brewery.weight-delta.v1".into(),
            parent_sha256: sha256.into(),
            revision_sha256: revision,
            changed_tensors: tensors.len(),
            changed_bytes,
            delta_bytes: delta.metadata()?.len(),
            delta_sha256: file_sha256(&mut delta)?,
        },
        tensors,
    })
}

fn finite(dtype: &str, bytes: &[u8]) -> bool {
    if dtype == "F32" {
        f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()
    } else {
        let mask = if dtype == "F16" { 0x7c00 } else { 0x7f80 };
        u16::from_le_bytes(bytes.try_into().unwrap()) & mask != mask
    }
}

// SplitMix64, counter-based: use one sample per ordered input and base byte
// offset. Skipping equal blocks never changes the mask on other blocks.
fn uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 * (1. / (1u64 << 53) as f64)
}

#[derive(Clone, Copy)]
enum Trim {
    All,
    None,
    Cut { bits: u32, equal_left: u64 },
}

impl Trim {
    fn apply(&mut self, value: f32) -> Result<f32> {
        ensure!(value.is_finite(), "non-finite merge delta");
        match self {
            Self::All => Ok(value),
            Self::None => Ok(0.),
            Self::Cut { bits, equal_left } => match value.abs().to_bits().cmp(bits) {
                std::cmp::Ordering::Greater => Ok(value),
                std::cmp::Ordering::Equal if *equal_left > 0 => {
                    *equal_left -= 1;
                    Ok(value)
                }
                _ => Ok(0.),
            },
        }
    }
}

// Radix selection finds the kth largest nonnegative F32 bit pattern without
// materializing the tensor. At the cutoff keep equal values by flat index.
#[allow(clippy::too_many_arguments)]
fn cutoff(
    base: &File,
    base_offset: u64,
    peer: &File,
    peer_offset: u64,
    size: u64,
    dtype: &str,
    width: usize,
    mut k: u64,
    a: &mut [u8],
    b: &mut [u8],
) -> Result<Trim> {
    let mut prefix = 0;
    let mut mask = 0;
    for shift in [24, 16, 8, 0] {
        let mut counts = [0; 256];
        let mut nonzero = 0;
        for pos in (0..size).step_by(BLOCK) {
            let n = (size - pos).min(BLOCK as u64) as usize;
            base.read_exact_at(&mut a[..n], base_offset + pos)?;
            peer.read_exact_at(&mut b[..n], peer_offset + pos)?;
            if a[..n] == b[..n] {
                if prefix == 0 {
                    counts[0] += (n / width) as u64;
                }
                continue;
            }
            for (a, b) in a[..n].chunks_exact(width).zip(b[..n].chunks_exact(width)) {
                let d = decode(dtype, b) as f32 - decode(dtype, a) as f32;
                ensure!(d.is_finite(), "non-finite merge delta");
                let bits = d.abs().to_bits();
                nonzero += u64::from(bits != 0);
                if bits & mask == prefix {
                    counts[((bits >> shift) & 255) as usize] += 1;
                }
            }
        }
        // A zero cutoff keeps all nonzero deltas regardless of which zeros tie.
        if shift == 24 && nonzero <= k {
            return Ok(Trim::All);
        }
        let selected = counts
            .iter()
            .enumerate()
            .rfind(|&(_, count)| {
                if k > *count {
                    k -= *count;
                    false
                } else {
                    true
                }
            })
            .context("could not select merge cutoff")?
            .0 as u32;
        prefix |= selected << shift;
        mask |= 255 << shift;
    }
    Ok(Trim::Cut {
        bits: prefix,
        equal_left: k,
    })
}

#[cfg(test)]
mod tests;
