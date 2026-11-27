//! Native-store projection for canonical HGN v2 checkpoints.
mod audit;
use super::{TensorInfo, file_sha256, open_regular};
use anyhow::{Context, Result, ensure};
pub use audit::audit;
use bf16_to_hgn::{
    ENCODER_SHA256,
    codec::{EncodedBlock, Encoding, QuantMeta, decode_rows, encode_rows, payload_layout},
    container::{ENTRY_SIZE, Entry, HEADER_SIZE, Reader, fingerprint},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    ops::RangeInclusive,
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::{Path, PathBuf},
    thread,
};

pub const MODEL: &str = "model.hgn";
pub const REPORT: &str = "hgn-feedback.json";
const SCHEMA: &str = "brewery.hgn-feedback.v1";
const MAX_MATRIX_ELEMENTS: usize = 16 << 20;
const MAX_STEPS: usize = 32;
const CACHE_HEADER: u64 = 4096;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorReport {
    pub store: u32,
    pub variant: u32,
    pub bytes: u64,
    pub shape: Vec<u64>,
    /// Maximum number of encode passes on any expert in this tensor.
    pub iterations: usize,
    pub tolerance: f64,
    pub parent_mean_abs_component: f64,
    /// Component after one native encode, divided by the parent component.
    pub leftover_before: f64,
    /// Component in the final stored payload, divided by the parent component.
    pub leftover_after: f64,
    /// Frobenius norm of (stored - parent), divided by the parent norm.
    pub rel_error: f64,
    pub payload_sha256: String,
}

impl TensorReport {
    pub fn info(&self, name: String) -> TensorInfo {
        TensorInfo {
            name,
            shape: self.shape.clone(),
            dtype: match (self.store, self.variant) {
                (0, 0) => "BF16",
                (5, 2) => "Q4CP",
                (7, 0) => "Q8G64",
                (16, 0x1206) => "HT3",
                (16, 0x1208) => "HT4",
                (23, 128) => "I4R",
                _ => "unknown",
            }
            .into(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema: String,
    pub encoder_sha256: String,
    pub parent_sha256: String,
    pub revision_sha256: String,
    pub direction: Vec<f64>,
    pub tensors: BTreeMap<String, TensorReport>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Experts,
    Dense,
    Small,
}

struct Writer {
    index: usize,
    entry: Entry,
    experts: usize,
    rows: usize,
    cols: usize,
    part: Part,
}

fn policy(encoding: Encoding) -> Result<(f64, f64)> {
    Ok(match encoding {
        Encoding::Ht | Encoding::Ht3 => (0.5, 0.08),
        Encoding::I4r | Encoding::Q4Cp | Encoding::Q8G64 => (1.0, 0.05),
        // Compute the exact projection before rounding to BF16. Measure the
        // actual rounding residue on decode rather than reporting a fake zero.
        Encoding::Bf16 => (0.0, 0.01),
        _ => anyhow::bail!("unsupported HGN residual writer store: {encoding:?}"),
    })
}

fn writers(reader: &Reader, layers: RangeInclusive<usize>, width: usize) -> Result<Vec<Writer>> {
    let mut selected = Vec::new();
    let entries = &reader.layout.entries;
    for layer in layers {
        let root = format!("layers.{layer}.");
        let mut attention = false;
        let mut ffn = false;
        let mut shared = false;
        let mut routed = false;
        for (index, entry) in entries.iter().enumerate() {
            let Some(stem) = entry.name.strip_prefix(&root) else {
                continue;
            };
            let (flag, stacked) = match stem {
                "self_attn.o_proj.weight" | "linear_attn.out_proj.weight" => {
                    (&mut attention, false)
                }
                "mlp.down_proj.weight" => (&mut ffn, false),
                "mlp.shared_expert.down_proj.weight" => (&mut shared, false),
                "mlp.experts.down_proj.weight" => (&mut routed, true),
                "ple.value_proj.weight" => {
                    selected.push(writer(index, entry, false, width)?);
                    continue;
                }
                _ => continue,
            };
            ensure!(!*flag, "ambiguous residual writer in layer {layer}: {stem}");
            *flag = true;
            selected.push(writer(index, entry, stacked, width)?);
        }
        // A model can have no shared expert. If it declares routed experts,
        // do not silently edit a shared writer without its routed partner.
        let expert_prefix = format!("{root}mlp.experts.");
        let has_experts = entries.iter().any(|e| e.name.starts_with(&expert_prefix));
        ensure!(
            attention && (ffn || routed) && (!shared || routed) && (!has_experts || routed),
            "layer {layer} is missing a supported attention or FFN residual writer"
        );
    }
    for w in &selected {
        let stem = w.entry.name.strip_suffix(".weight").unwrap();
        let bias = format!("{stem}.bias");
        ensure!(
            !entries.iter().any(|e| e.name == bias),
            "ablation does not support biased residual writer {stem}"
        );
    }
    Ok(selected)
}

fn writer(index: usize, entry: &Entry, stacked: bool, width: usize) -> Result<Writer> {
    let (experts, rows, cols) = match (stacked, entry.shape.as_slice()) {
        (false, [rows, cols]) => (1, *rows, *cols),
        (true, [experts @ 1..=512, rows, cols]) => (*experts, *rows, *cols),
        _ => anyhow::bail!("invalid HGN residual writer rank: {}", entry.name),
    };
    let (rows, cols) = (usize::try_from(rows)?, usize::try_from(cols)?);
    ensure!(
        rows == width && cols <= MAX_MATRIX_ELEMENTS / rows,
        "HGN residual writer shape does not match the direction or exceeds the matrix limit: {}",
        entry.name
    );
    policy(entry.encoding)?;
    Ok(Writer {
        index,
        entry: entry.clone(),
        experts: experts as usize,
        rows,
        cols,
        part: if stacked {
            Part::Experts
        } else if entry.encoding.ht_bits().is_some() {
            Part::Dense
        } else {
            Part::Small
        },
    })
}

/// Build a private child; an interrupted run can reuse completed tensor
/// payloads under cache_root. Output is never installed if a writer misses
/// its store tolerance. The caller owns the private revision directory.
pub fn build(
    source: &Path,
    expected: &str,
    output: &Path,
    cache_root: &Path,
    layers: RangeInclusive<usize>,
    direction: &[f32],
    disk_reserve: u64,
) -> Result<Report> {
    ensure!(
        !layers.is_empty() && *layers.end() < 4096 && layers.end() - layers.start() < 64,
        "select an inclusive range of at most 64 layers from 0 to 4095"
    );
    ensure!(
        (1..=16384).contains(&direction.len()) && direction.iter().all(|x| x.is_finite()),
        "invalid HGN direction width or values"
    );
    let norm = direction
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(norm > 0.0, "HGN direction is zero");
    let direction: Vec<_> = direction.iter().map(|&x| f64::from(x) / norm).collect();
    ensure!(!output.exists(), "HGN child already exists");
    let mut source_file = open_regular(source)?;
    ensure!(crate::is_digest(expected), "invalid parent HGN SHA-256");
    let dir = output
        .parent()
        .context("HGN child has no output directory")?;
    let bytes = source_file.metadata()?.len();
    check_disk(dir, disk_reserve, bytes)?;
    let mut child = tempfile::NamedTempFile::new_in(dir)?;
    ensure!(
        io::copy(
            &mut (&mut source_file).take(bytes.saturating_add(1)),
            child.as_file_mut()
        )? == bytes
            && file_sha256(child.as_file_mut())? == expected,
        "parent HGN SHA-256 mismatch"
    );
    // All the codec workers read the same private snapshot of the pinned bytes.
    let snapshot = child.path();
    let reader = Reader::open(snapshot)?;
    ensure!(reader.version == 2, "expected native HGN v2");
    let selected = writers(&reader, layers.clone(), direction.len())?;
    private_dir(cache_root)?;
    let key = digest(&serde_json::to_vec(&(
        SCHEMA,
        ENCODER_SHA256,
        expected,
        layers,
        &direction,
    ))?);
    let cache = cache_root.join(key);
    private_dir(&cache)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(cache.join("lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock).context("HGN edit cache is busy")?;
    // All cache paths are generator-owned and live beneath a private directory.
    let mut reports = BTreeMap::new();
    let mut missing = Vec::new();
    for w in &selected {
        // A crash can leave a partial file. It is never read as cache evidence.
        remove_if_exists(&cache.join(format!("{}.part", w.index)))?;
        let path = cache_path(&cache, w.index);
        if let Ok((_, report, _)) = cached(&path, w) {
            reports.insert(w.entry.name.clone(), report);
        } else {
            remove_if_exists(&path)?;
            missing.push(w);
        }
    }
    let missing_bytes: u64 = missing.iter().map(|w| w.entry.size + CACHE_HEADER).sum();
    check_disk(&cache, disk_reserve, missing_bytes)?;
    thread::scope(|scope| {
        let tasks: Vec<_> = [Part::Experts, Part::Dense, Part::Small]
            .into_iter()
            .filter_map(|part| {
                let work: Vec<_> = missing.iter().copied().filter(|w| w.part == part).collect();
                if work.is_empty() {
                    return None;
                }
                let (cache, direction) = (&cache, &direction);
                Some(scope.spawn(move || -> Result<_> {
                    let mut reader = Reader::open(snapshot)?;
                    work.into_iter()
                        .map(|w| {
                            encode_to_cache(&mut reader, cache, w, direction)
                                .with_context(|| format!("HGN writer {}", w.entry.name))
                                .map(|report| (w.entry.name.clone(), report))
                        })
                        .collect::<Result<Vec<_>>>()
                }))
            })
            .collect();
        let mut result = Ok(());
        for task in tasks {
            match task
                .join()
                .map_err(|_| anyhow::anyhow!("HGN edit part panicked"))?
            {
                Ok(part) => reports.extend(part),
                Err(err) => result = Err(err),
            }
        }
        result
    })?;

    for w in &selected {
        let (mut file, _, xor32) = cached(&cache_path(&cache, w.index), w)?;
        child.seek(SeekFrom::Start(w.entry.offset))?;
        ensure!(
            io::copy(&mut (&mut file).take(w.entry.size), child.as_file_mut())? == w.entry.size,
            "short cached HGN payload"
        );
        child.seek(SeekFrom::Start(
            HEADER_SIZE + w.index as u64 * ENTRY_SIZE + 152,
        ))?;
        child.write_all(&xor32.to_le_bytes())?;
    }
    child.as_file().sync_all()?;
    let revision_sha256 = file_sha256(child.as_file_mut())?;
    child.persist_noclobber(output)?;
    File::open(dir)?.sync_all()?;
    Ok(Report {
        schema: SCHEMA.into(),
        encoder_sha256: ENCODER_SHA256.into(),
        parent_sha256: expected.into(),
        revision_sha256,
        direction,
        tensors: reports,
    })
}

fn private_dir(path: &Path) -> Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_dir() && meta.mode() & 0o7777 == 0o700 && meta.uid() == unsafe { libc::geteuid() },
        "HGN cache directory must be private and owned by the editor"
    );
    Ok(())
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        status => status,
    }
}

fn check_disk(path: &Path, reserve: u64, bytes: u64) -> Result<()> {
    ensure!(
        crate::available(path)? >= reserve.saturating_add(bytes),
        "not enough disk space for HGN scratch and the configured reserve"
    );
    Ok(())
}

fn cache_path(cache: &Path, index: usize) -> PathBuf {
    cache.join(format!("{index}.payload"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn cached(path: &Path, w: &Writer) -> Result<(File, TensorReport, u32)> {
    let mut file = open_regular(path)?;
    ensure!(
        file.metadata()?.len() == CACHE_HEADER + w.entry.size,
        "cache size mismatch"
    );
    let mut header = [0; CACHE_HEADER as usize];
    file.read_exact(&mut header)?;
    let line = header
        .iter()
        .position(|&b| b == b'\n')
        .context("missing cache receipt")?;
    let report: TensorReport = serde_json::from_slice(&header[..line])?;
    let (store, variant) = (w.entry.encoding.code(), w.entry.encoding.variant());
    ensure!(
        (report.store, report.variant, report.bytes) == (store, variant, w.entry.size)
            && report.shape == w.entry.shape
            && report.tolerance == policy(w.entry.encoding)?.1
            && report.leftover_after <= report.tolerance,
        "invalid HGN cached tensor receipt"
    );
    let f = fingerprint(&mut file, CACHE_HEADER, w.entry.size)?;
    ensure!(
        f.sha256 == report.payload_sha256,
        "cache payload SHA-256 mismatch"
    );
    file.seek(SeekFrom::Start(CACHE_HEADER))?;
    Ok((file, report, f.xor32))
}

fn side_plane(reader: &mut Reader, name: &str, len: usize) -> Result<Vec<f32>> {
    let (index, entry) = reader
        .layout
        .entries
        .iter()
        .enumerate()
        .find(|(_, e)| e.name == name)
        .with_context(|| format!("missing HGN rotation plane {name}"))?;
    ensure!(
        entry.encoding == Encoding::F16 && entry.size == 2 * len as u64,
        "invalid HGN rotation plane {name}"
    );
    let checksum = entry.checksum;
    ensure!(
        reader.fingerprint(index)?.xor32 == checksum,
        "parent rotation plane checksum mismatch: {name}"
    );
    let codes = reader.read_at(index, 0, 2 * len)?;
    decode_rows(
        Encoding::F16,
        &EncodedBlock {
            codes,
            scales: vec![],
        },
        1,
        len,
        &QuantMeta::default(),
    )
}

struct Blocks {
    code_start: u64,
    code_bytes: u64,
    scale_start: u64,
    scale_bytes: u64,
}

impl Blocks {
    fn new(w: &Writer) -> Result<Self> {
        let one = payload_layout(w.entry.encoding, w.rows, w.cols)?;
        let all = payload_layout(w.entry.encoding, w.experts * w.rows, w.cols)?;
        let planar = matches!(w.entry.encoding, Encoding::Q4Cp | Encoding::I4r);
        Ok(Self {
            code_start: all.codes_offset,
            code_bytes: if planar {
                (w.rows * w.cols / 2) as u64
            } else {
                one.len
            },
            scale_start: all.scales_offset,
            scale_bytes: if planar {
                w.rows as u64 * one.scale_row_stride
            } else {
                0
            },
        })
    }

    fn read(&self, reader: &mut Reader, index: usize, expert: usize) -> Result<EncodedBlock> {
        let e = expert as u64;
        Ok(EncodedBlock {
            codes: reader.read_at(
                index,
                self.code_start + e * self.code_bytes,
                self.code_bytes as usize,
            )?,
            scales: if self.scale_bytes == 0 {
                vec![]
            } else {
                reader.read_at(
                    index,
                    self.scale_start + e * self.scale_bytes,
                    self.scale_bytes as usize,
                )?
            },
        })
    }

    fn write(&self, file: &mut File, expert: usize, block: &EncodedBlock) -> Result<()> {
        let e = expert as u64;
        ensure!(
            block.codes.len() as u64 == self.code_bytes
                && block.scales.len() as u64 == self.scale_bytes,
            "HGN codec changed payload size"
        );
        file.seek(SeekFrom::Start(
            CACHE_HEADER + self.code_start + e * self.code_bytes,
        ))?;
        file.write_all(&block.codes)?;
        if self.scale_bytes != 0 {
            file.seek(SeekFrom::Start(
                CACHE_HEADER + self.scale_start + e * self.scale_bytes,
            ))?;
            file.write_all(&block.scales)?;
        }
        Ok(())
    }
}

fn encode_to_cache(
    reader: &mut Reader,
    cache: &Path,
    w: &Writer,
    r: &[f64],
) -> Result<TensorReport> {
    let (index, enc, rows, cols) = (w.index, w.entry.encoding, w.rows, w.cols);
    ensure!(
        reader.fingerprint(index)?.xor32 == w.entry.checksum,
        "parent tensor checksum mismatch"
    );
    let layout = Blocks::new(w)?;
    let (gain, tolerance) = policy(enc)?;
    let mut codebook = [0.; 16];
    let mut codebook_bytes = vec![];
    if enc == Encoding::Q4Cp {
        codebook_bytes = reader.read_at(index, 0, 64)?;
        for (value, bytes) in codebook.iter_mut().zip(codebook_bytes.chunks_exact(4)) {
            *value = f32::from_le_bytes(bytes.try_into()?);
        }
    }
    let mut su = Vec::new();
    let mut sv = Vec::new();
    let ht = enc.ht_bits().is_some();
    let stacked_ht = ht && w.part == Part::Experts;
    if ht || enc == Encoding::I4r {
        let root = w.entry.name.strip_suffix(".weight").unwrap();
        let (su_count, sv_count) = if stacked_ht {
            (w.experts * cols, w.experts * rows)
        } else {
            (cols, rows)
        };
        su = side_plane(reader, &format!("{root}.suh"), su_count)?;
        sv = side_plane(reader, &format!("{root}.svh"), sv_count)?;
    }
    let partial = cache.join(format!("{index}.part"));
    let mut file = File::create_new(&partial)?;
    file.set_len(CACHE_HEADER + w.entry.size)?;
    file.seek(SeekFrom::Start(CACHE_HEADER))?;
    file.write_all(&codebook_bytes)?;

    let mut total = Metrics::default();
    let threads = if ht {
        thread::available_parallelism().map_or(1, |n| ((n.get() + 1) / 2).clamp(1, 8))
    } else {
        1
    };
    for e in 0..w.experts {
        let (o, k) = if stacked_ht {
            (e * rows, e * cols)
        } else {
            (0, 0)
        };
        let meta = QuantMeta {
            su: su.get(k..k + cols).unwrap_or_default(),
            sv: sv.get(o..o + rows).unwrap_or_default(),
            codebook: &codebook,
            threads,
            ..QuantMeta::default()
        };
        let stock = layout.read(reader, index, e)?;
        let (payload, stats) = feedback(enc, stock, rows, cols, &meta, r, gain, tolerance)?;
        layout.write(&mut file, e, &payload)?;
        total.add(stats);
    }
    let report = TensorReport {
        store: enc.code(),
        variant: enc.variant(),
        shape: w.entry.shape.clone(),
        bytes: w.entry.size,
        iterations: total.iterations,
        tolerance,
        parent_mean_abs_component: total.parent / (w.experts * cols) as f64,
        leftover_before: ratio(total.first, total.parent),
        leftover_after: ratio(total.last, total.parent),
        rel_error: ratio(total.diff_sq.sqrt(), total.parent_sq.sqrt()),
        payload_sha256: fingerprint(&mut file, CACHE_HEADER, w.entry.size)?.sha256,
    };
    if report.leftover_after > tolerance {
        remove_if_exists(&partial)?;
        anyhow::bail!(
            "decoded HGN writer did not converge: leftover {:.6} -> {:.6}, tolerance {tolerance}",
            report.leftover_before,
            report.leftover_after,
        );
    }
    let mut header = serde_json::to_vec(&report)?;
    header.push(b'\n');
    ensure!(
        header.len() as u64 <= CACHE_HEADER,
        "oversized HGN cache receipt"
    );
    file.rewind()?;
    file.write_all(&header)?;
    file.sync_all()?;
    fs::rename(&partial, cache_path(cache, index))?;
    File::open(cache)?.sync_all()?;
    Ok(report)
}

fn decode(
    enc: Encoding,
    block: &EncodedBlock,
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<Vec<f32>> {
    if let Some(bits) = enc.ht_bits() {
        let band = 128 * cols * bits / 8;
        let mut values = Vec::with_capacity(rows * cols);
        for (i, codes) in block.codes.chunks_exact(band).enumerate() {
            let m = QuantMeta {
                sv: &meta.sv[i * 128..(i + 1) * 128],
                ..*meta
            };
            let b = EncodedBlock {
                codes: codes.to_vec(),
                scales: vec![],
            };
            values.extend(decode_rows(enc, &b, 128, cols, &m)?);
        }
        Ok(values)
    } else {
        decode_rows(enc, block, rows, cols, meta)
    }
}

#[derive(Default)]
struct Metrics {
    iterations: usize,
    parent: f64,
    first: f64,
    last: f64,
    parent_sq: f64,
    diff_sq: f64,
}

impl Metrics {
    fn add(&mut self, other: Self) {
        self.iterations = self.iterations.max(other.iterations);
        self.parent += other.parent;
        self.first += other.first;
        self.last += other.last;
        self.parent_sq += other.parent_sq;
        self.diff_sq += other.diff_sq;
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn components(values: &[f32], r: &[f64], cols: usize) -> Vec<f64> {
    let mut result = vec![0.; cols];
    for (&d, row) in r.iter().zip(values.chunks_exact(cols)) {
        for (c, &v) in result.iter_mut().zip(row) {
            *c += d * f64::from(v);
        }
    }
    result
}

fn subtract(target: &mut [f32], r: &[f64], c: &[f64], gain: f64) {
    for (&d, row) in r.iter().zip(target.chunks_exact_mut(c.len())) {
        for (v, &p) in row.iter_mut().zip(c) {
            *v = (f64::from(*v) - gain * d * p) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn feedback(
    enc: Encoding,
    stock: EncodedBlock,
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
    r: &[f64],
    gain: f64,
    tolerance: f64,
) -> Result<(EncodedBlock, Metrics)> {
    let original = decode(enc, &stock, rows, cols, meta)?;
    ensure!(
        original.iter().all(|x| x.is_finite()),
        "nonfinite parent HGN weights"
    );
    let c0 = components(&original, r, cols);
    let parent: f64 = c0.iter().map(|x| x.abs()).sum();
    let mut stats = Metrics {
        parent,
        parent_sq: original.iter().map(|&x| f64::from(x).powi(2)).sum(),
        last: f64::INFINITY,
        ..Metrics::default()
    };
    if parent == 0.0 {
        stats.last = 0.0;
        return Ok((stock, stats));
    }
    let mut target = original.clone();
    subtract(&mut target, r, &c0, 1.0);
    let mut best = stock;
    for i in 1..=if gain == 0.0 { 1 } else { MAX_STEPS } {
        let payload = encode_rows(enc, &target, rows, cols, meta)?;
        let stored = decode(enc, &payload, rows, cols, meta)?;
        ensure!(
            stored.iter().all(|x| x.is_finite()),
            "nonfinite edited HGN weights"
        );
        let c = components(&stored, r, cols);
        let leftover: f64 = c.iter().map(|x| x.abs()).sum();
        stats.iterations = i;
        if i == 1 {
            stats.first = leftover;
        }
        if leftover < stats.last {
            stats.last = leftover;
            stats.diff_sq = stored
                .iter()
                .zip(&original)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            best = payload;
        }
        if leftover <= tolerance * parent {
            break;
        }
        subtract(&mut target, r, &c, gain);
    }
    Ok((best, stats))
}
