//! Compare pinned scratch files without loading a checkpoint into memory.
use super::*;
use std::io::BufReader;

const BLOCK: usize = 64 << 10;
const MAX_ROWS: usize = 4096;
const SAMPLE: u64 = 32;

pub struct Delta<'a> {
    pub file: &'a mut File,
    pub summary: &'a Summary,
}

#[derive(Serialize)]
pub struct TensorDiff {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    changed_elements: u64,
    bounds: Bounds,
    l2: f64,
    linf: f64,
    relative_frobenius: Option<f64>,
    low_rank: Option<Spectrum>,
}

#[derive(Serialize)]
struct Bounds {
    start: Vec<u64>,
    end: Vec<u64>,
}

#[derive(Serialize)]
struct Spectrum {
    sample_shape: [u64; 2],
    singular_values: Vec<f64>,
    sample_rank: usize,
}

pub fn compare(
    a: &mut File,
    a_sha: &str,
    b: &mut File,
    b_sha: &str,
    delta: Option<Delta<'_>>,
) -> Result<Vec<TensorDiff>> {
    let a_size = pinned(a, a_sha, MAX_FILE)?;
    let b_size = pinned(b, b_sha, MAX_FILE)?;
    ensure!(a_size == b_size, "revision sizes differ");
    let ah = Header::read(&mut *a, a_size, super::super::LIMIT)?;
    let bh = Header::read(&mut *b, b_size, super::super::LIMIT)?;
    ensure!(
        ah.payload_start == bh.payload_start
            && ah.entries.len() == bh.entries.len()
            && ah
                .entries
                .iter()
                .all(|(name, x)| bh.entries.get(name).is_some_and(|y| {
                    x.dtype == y.dtype && x.shape == y.shape && x.data_offsets == y.data_offsets
                })),
        "revision tensor layouts differ"
    );
    let names = delta
        .map(|d| candidates(d, a_size, a_sha, b_sha, &ah, a, b))
        .transpose()?;
    let mut rows = Vec::new();
    if a_sha == b_sha {
        return Ok(rows);
    }
    let mut left = vec![0; BLOCK];
    let mut right = vec![0; BLOCK];
    for (name, t) in ah.entries {
        let start = ah.payload_start + t.data_offsets[0];
        let size = t.data_offsets[1] - t.data_offsets[0];
        if let Some(names) = &names {
            if !names.contains(&name) {
                continue;
            }
        } else {
            let mut changed = false;
            read_pair(a, b, start, size, &mut left, &mut right, |_, a, b| {
                changed = a != b;
                Ok(!changed)
            })?;
            if !changed {
                continue;
            }
        }
        ensure!(rows.len() < MAX_ROWS, "too many changed tensors");
        rows.push(stats(name, t, start, a, b, &mut left, &mut right)?);
    }
    Ok(rows)
}

fn pinned(file: &mut File, hash: &str, max: u64) -> Result<u64> {
    file.rewind()?;
    let size = file.metadata()?.len();
    ensure!(size <= max, "revision file exceeds diff limit");
    let mut sha = Sha256::new();
    let read = io::copy(&mut (&mut *file).take(size + 1), &mut sha)?;
    ensure!(
        read == size && format!("{:x}", sha.finalize()) == hash,
        "revision file does not match its pinned SHA-256"
    );
    file.rewind()?;
    Ok(size)
}

fn candidates(
    delta: Delta<'_>,
    size: u64,
    a_sha: &str,
    b_sha: &str,
    header: &Header,
    a: &mut File,
    b: &mut File,
) -> Result<std::collections::BTreeSet<String>> {
    let s = delta.summary;
    ensure!(
        s.schema == "brewery.weight-delta.v1"
            && ((s.parent_sha256 == a_sha && s.revision_sha256 == b_sha)
                || (s.parent_sha256 == b_sha && s.revision_sha256 == a_sha))
            && pinned(delta.file, &s.delta_sha256, MAX_EDIT_BYTES + DELTA_OVERHEAD)?
                == s.delta_bytes,
        "scratch delta metadata does not match the revisions"
    );
    let mut file = BufReader::new(delta.file);
    fn read<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
        let mut bytes = [0; N];
        r.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    ensure!(
        &read::<8>(&mut file)? == MAGIC
            && read::<64>(&mut file)? == s.parent_sha256.as_bytes()
            && read::<64>(&mut file)? == s.revision_sha256.as_bytes()
            && u64::from_le_bytes(read(&mut file)?) == size,
        "scratch delta header does not match the revisions"
    );
    let count = u32::from_le_bytes(read(&mut file)?) as usize;
    ensure!(
        count <= 32 && count == s.changed_tensors,
        "invalid scratch delta count"
    );
    let mut changed_bytes = 0;
    let mut last = header.payload_start;
    let mut names = std::collections::BTreeSet::new();
    let mut left = vec![0; BLOCK];
    let mut right = vec![0; BLOCK];
    let mut xor = vec![0; BLOCK];
    for _ in 0..count {
        let offset = u64::from_le_bytes(read(&mut file)?);
        let len = u64::from_le_bytes(read(&mut file)?);
        ensure!(offset >= last && len > 0, "invalid scratch XOR range");
        last = offset
            .checked_add(len)
            .context("scratch XOR range overflow")?;
        let (name, _) = header
            .entries
            .iter()
            .find(|(_, t)| {
                let [start, end] = t.data_offsets;
                offset >= header.payload_start + start && last <= header.payload_start + end
            })
            .context("scratch XOR range crosses a tensor boundary")?;
        ensure!(names.insert(name.clone()), "duplicate scratch XOR tensor");
        let mut different = 0u64;
        read_pair(a, b, offset, len, &mut left, &mut right, |_, a, b| {
            file.read_exact(&mut xor[..a.len()])?;
            for ((&x, &y), &d) in a.iter().zip(b).zip(&xor[..a.len()]) {
                ensure!(x ^ y == d, "scratch XOR data differs from pinned weights");
                different += u64::from(d != 0);
            }
            Ok(true)
        })?;
        ensure!(different > 0, "empty scratch XOR record");
        changed_bytes += different;
    }
    ensure!(
        changed_bytes == s.changed_bytes
            && file.read(&mut xor[..1])? == 0
            && (count == 0) == (a_sha == b_sha),
        "scratch XOR inventory does not match its summary"
    );
    Ok(names)
}

fn read_pair(
    a: &mut File,
    b: &mut File,
    start: u64,
    mut size: u64,
    left: &mut [u8],
    right: &mut [u8],
    mut each: impl FnMut(u64, &[u8], &[u8]) -> Result<bool>,
) -> Result<()> {
    a.seek(SeekFrom::Start(start))?;
    b.seek(SeekFrom::Start(start))?;
    let mut offset = 0;
    while size != 0 {
        let n = size.min(left.len() as u64) as usize;
        a.read_exact(&mut left[..n])?;
        b.read_exact(&mut right[..n])?;
        if !each(offset, &left[..n], &right[..n])? {
            break;
        }
        offset += n as u64;
        size -= n as u64;
    }
    Ok(())
}

fn stats(
    name: String,
    t: Tensor,
    start: u64,
    a: &mut File,
    b: &mut File,
    left: &mut [u8],
    right: &mut [u8],
) -> Result<TensorDiff> {
    let width = editable_width(&t.dtype)?;
    let mut bounds = Bounds {
        start: t.shape.clone(),
        end: vec![0; t.shape.len()],
    };
    let (mut base2, mut delta2, mut linf, mut changed) = (0., 0., 0f64, 0);
    let size = t.data_offsets[1] - t.data_offsets[0];
    read_pair(a, b, start, size, left, right, |offset, a, b| {
        for (i, (a, b)) in a.chunks_exact(width).zip(b.chunks_exact(width)).enumerate() {
            let x = decode(&t.dtype, a);
            let y = decode(&t.dtype, b);
            ensure!(x.is_finite() && y.is_finite(), "non-finite tensor value");
            base2 += x * x;
            if a != b {
                let d = y - x;
                delta2 += d * d;
                linf = linf.max(d.abs());
                changed += 1;
                let mut index = offset / width as u64 + i as u64;
                for (axis, &dim) in t.shape.iter().enumerate().rev() {
                    let coordinate = index % dim;
                    index /= dim;
                    bounds.start[axis] = bounds.start[axis].min(coordinate);
                    bounds.end[axis] = bounds.end[axis].max(coordinate + 1);
                }
            }
        }
        Ok(true)
    })?;
    ensure!(changed > 0, "tensor does not match scratch delta inventory");
    let low_rank = if let [_, columns] = t.shape.as_slice() {
        let rows = sampled(bounds.start[0], bounds.end[0]);
        let cols = sampled(bounds.start[1], bounds.end[1]);
        let mut sample = vec![vec![0.; cols.len()]; rows.len()];
        for (i, row) in rows.into_iter().enumerate() {
            for (j, &col) in cols.iter().enumerate() {
                read_pair(
                    a,
                    b,
                    start + (row * columns + col) * width as u64,
                    width as u64,
                    left,
                    right,
                    |_, a, b| {
                        sample[i][j] = decode(&t.dtype, b) - decode(&t.dtype, a);
                        Ok(true)
                    },
                )?;
            }
        }
        Some(spectrum(&mut sample))
    } else {
        None
    };
    Ok(TensorDiff {
        name,
        shape: t.shape,
        dtype: t.dtype,
        changed_elements: changed,
        bounds,
        l2: delta2.sqrt(),
        linf,
        relative_frobenius: (base2 > 0.).then(|| (delta2 / base2).sqrt()),
        low_rank,
    })
}

fn sampled(start: u64, end: u64) -> Vec<u64> {
    let n = (end - start).min(SAMPLE);
    (0..n)
        .map(|i| start + i * (end - start - 1) / (n - 1).max(1))
        .collect()
}

// One-sided Jacobi SVD on the sampled submatrix. Report values in sample units.
fn spectrum(sample: &mut [Vec<f64>]) -> Spectrum {
    let n = sample[0].len();
    for _ in 0..32 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (mut a, mut b, mut c) = (0., 0., 0.);
                for r in &*sample {
                    a += r[p] * r[p];
                    b += r[q] * r[q];
                    c += r[p] * r[q];
                }
                if c.abs() <= 1e-12 * (a * b).sqrt() {
                    continue;
                }
                let tau = (b - a) / (2. * c);
                let t = tau.signum() / (tau.abs() + tau.hypot(1.));
                let cos = 1. / t.hypot(1.);
                let sin = t * cos;
                for r in &mut *sample {
                    (r[p], r[q]) = (cos * r[p] - sin * r[q], sin * r[p] + cos * r[q]);
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }
    let mut values: Vec<_> = (0..n)
        .map(|j| sample.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt())
        .collect();
    values.sort_by(|a, b| b.total_cmp(a));
    values.truncate(n.min(sample.len()));
    let sample_rank = values.iter().filter(|&&s| s > values[0] * 1e-6).count();
    values.truncate(8);
    Spectrum {
        sample_shape: [sample.len() as u64, n as u64],
        singular_values: values,
        sample_rank,
    }
}
