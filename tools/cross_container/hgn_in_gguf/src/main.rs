use anyhow::{bail, ensure, Context, Result};
use bf16_to_hgn::{
    codec::{self, EncodedBlock, Encoding, QuantMeta},
    hgn_source::HgnSidecar,
    source::TensorSource,
};
use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env,
    fs::{self, OpenOptions},
    io::{Cursor, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};

const SCHEMA: &str = "brewery.cross-container.v1";
const LIMIT: u64 = 64 << 20;
const TYPE_BASE: u32 = 0x4252_0100;
const ALIGN: usize = 32;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read(path: &Path) -> Result<Vec<u8>> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let stat = file.metadata()?;
    ensure!(
        stat.is_file() && stat.len() <= LIMIT,
        "requires a bounded regular file (64 MiB)"
    );
    let stamp = |s: &fs::Metadata| {
        (
            s.dev(),
            s.ino(),
            s.len(),
            s.mtime(),
            s.mtime_nsec(),
            s.ctime(),
            s.ctime_nsec(),
        )
    };
    let mut out = Vec::new();
    Read::by_ref(&mut file)
        .take(LIMIT + 1)
        .read_to_end(&mut out)?;
    ensure!(
        out.len() as u64 == stat.len()
            && stamp(&stat) == stamp(&file.metadata()?)
            && stamp(&stat) == stamp(&fs::symlink_metadata(path)?),
        "input changed while reading"
    );
    Ok(out)
}

fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn u64le(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn text(b: &[u8]) -> Result<&str> {
    let n = b.iter().position(|x| *x == 0).unwrap_or(b.len());
    ensure!(n > 0 && b[n..].iter().all(|x| *x == 0), "invalid HGN text");
    let value = std::str::from_utf8(&b[..n])?;
    ensure!(
        value.bytes().all(|x| x.is_ascii_graphic()),
        "invalid HGN text"
    );
    Ok(value)
}

#[derive(Debug)]
struct Tensor<'a> {
    name: &'a str,
    shape: Vec<u64>,
    store: u32,
    variant: u32,
    offset: usize,
    checksum: u32,
    bytes: &'a [u8],
}

impl Tensor<'_> {
    fn encoding(&self) -> Option<Encoding> {
        match (self.store, self.variant) {
            (16, 0x1206) => Some(Encoding::Ht3),
            (16, 0x1208) => Some(Encoding::Ht),
            (23, 128) => Some(Encoding::I4r),
            _ => None,
        }
    }

    fn gguf_type(&self) -> u32 {
        match self.encoding() {
            Some(Encoding::Ht3) => TYPE_BASE + 1,
            Some(Encoding::Ht) => TYPE_BASE + 2,
            Some(Encoding::I4r) => TYPE_BASE + 3,
            _ => 1, // Standard GGML_TYPE_F16, only for the two companions.
        }
    }

    fn receipt(&self) -> Value {
        json!({"name":self.name,"shape":self.shape,"hgn_store":self.store,
            "hgn_variant":self.variant,"hgn_offset":self.offset,
            "gguf_type":self.gguf_type(),"xor32":self.checksum,
            "bytes":self.bytes.len(),"sha256":hash(self.bytes)})
    }
}

fn selection<'a>(source: &'a [u8], requested: Option<&str>) -> Result<([Tensor<'a>; 3], Value)> {
    ensure!(
        source.len() >= 104 && &source[..4] == b"HGN1" && u32le(source, 4) == 2,
        "requires HGN version 2"
    );
    let count = usize::try_from(u64le(source, 8))?;
    let table = usize::try_from(u64le(source, 16))?;
    let data = usize::try_from(u64le(source, 24))?;
    ensure!(
        (3..=4096).contains(&count)
            && table >= 104
            && table
                .checked_add(160 * count)
                .is_some_and(|end| end <= data)
            && data <= source.len()
            && data % 64 == 0
            && u64le(source, 32) == source.len() as u64,
        "invalid HGN inventory bounds"
    );
    let mut names = BTreeSet::new();
    let mut spans = Vec::with_capacity(count);
    let mut tensors = Vec::with_capacity(count);
    for row in source[table..table + 160 * count].as_chunks::<160>().0 {
        let name = text(&row[..96])?;
        ensure!(names.insert(name), "duplicate HGN tensor");
        let rank = u32le(row, 100) as usize;
        ensure!((1..=4).contains(&rank), "invalid HGN rank");
        let shape: Vec<_> = (0..4).map(|i| u64le(row, 104 + 8 * i)).collect();
        ensure!(
            shape[..rank]
                .iter()
                .all(|d| (1..=i64::MAX as u64).contains(d))
                && shape[rank..].iter().all(|d| *d == 0),
            "invalid HGN shape"
        );
        let offset = usize::try_from(u64le(row, 136))?;
        let size = usize::try_from(u64le(row, 144))?;
        let end = offset.checked_add(size).context("invalid HGN extent")?;
        ensure!(
            offset >= data && offset % 64 == 0 && size > 0 && end <= source.len(),
            "invalid HGN extent"
        );
        spans.push((offset, end));
        tensors.push(Tensor {
            name,
            shape: shape[..rank].into(),
            store: u32le(row, 96),
            variant: u32le(row, 156),
            offset,
            checksum: u32le(row, 152),
            bytes: &source[offset..end],
        });
    }
    spans.sort_unstable();
    ensure!(
        spans.windows(2).all(|r| r[0].1 <= r[1].0),
        "overlapping HGN tensors"
    );
    ensure!(
        requested.is_some() || count == 3,
        "complete export requires exactly three HGN inventory entries; use --tensor for a slice"
    );
    let weight = tensors
        .iter()
        .position(|t| t.encoding().is_some() && requested.is_none_or(|n| t.name == n))
        .context("no selected HT/I4R tensor")?;
    let weight = tensors.remove(weight);
    let base = weight
        .name
        .strip_suffix(".weight")
        .context("requires canonical HGN weight name")?;
    let enc = weight.encoding().unwrap();
    let (experts, rows, cols) = geometry(&weight.shape, enc)?;
    // This v1 slice supports rank-two HT and rank-three I4R down only. These
    // layouts share one SU vector and one SV vector across matrix instances.
    let su = format!("{base}.suh");
    let sv = format!("{base}.svh");
    let mut companion = |name: &str, len: usize| -> Result<Tensor<'a>> {
        let i = tensors
            .iter()
            .position(|t| t.name == name)
            .context("missing rotation companion")?;
        let t = tensors.remove(i);
        ensure!(
            t.store == 2 && t.variant == 0 && t.shape == [len as u64] && t.bytes.len() == len * 2,
            "rotation companion must be a matching FP16 vector"
        );
        Ok(t)
    };
    let su = companion(&su, cols)?;
    let sv = companion(&sv, rows)?;
    ensure!(
        weight.bytes.len() as u64 == codec::payload_layout(enc, experts * rows, cols)?.len,
        "wrong encoded weight payload length"
    );
    let selected = [weight, su, sv];
    for t in &selected {
        let xor = t.bytes.chunks(4).fold(0, |acc, chunk| {
            let mut b = [0; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            acc ^ u32::from_le_bytes(b)
        });
        ensure!(xor == t.checksum, "HGN checksum mismatch for {}", t.name);
    }
    let metadata = json!({
        "schema":SCHEMA,"scope":if requested.is_some() {"explicit_slice"} else {"complete_inventory"},
        "hgn":{"version":2,"identity":text(&source[40..104])?,"sha256":hash(source),
            "bytes":source.len(),"tensor_count":count,"table_offset":table,"data_offset":data,
            "header_table_sha256":hash(&[&source[..104], &source[table..table + count * 160]].concat())},
        "tensors":selected.iter().map(Tensor::receipt).collect::<Vec<_>>()
    });
    Ok((selected, metadata))
}

fn geometry(shape: &[u64], encoding: Encoding) -> Result<(usize, usize, usize)> {
    let (n, r, c) = match (encoding, shape) {
        (Encoding::Ht | Encoding::Ht3, &[r, c]) => (1, r, c),
        (Encoding::I4r, &[n, r, c]) => (n, r, c),
        _ => bail!("v1 requires rank-two HT or rank-three I4R"),
    };
    ensure!(
        n > 0
            && r > 0
            && c > 0
            && r % 128 == 0
            && c % 128 == 0
            && n.checked_mul(r)
                .and_then(|p| p.checked_mul(c))
                .is_some_and(|p| p <= 1 << 20),
        "unsupported or oversized rotated tensor geometry (1M elements)"
    );
    Ok((n as usize, r as usize, c as usize))
}

fn align(n: usize) -> usize {
    (n + ALIGN - 1) & !(ALIGN - 1)
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn gguf(tensors: &[Tensor<'_>; 3], metadata: &Value) -> Result<Vec<u8>> {
    let mut out = b"GGUF\x03\0\0\0".to_vec();
    out.extend_from_slice(&3u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());
    for (k, v) in [
        ("general.architecture", "brewery-hgn"),
        (SCHEMA, &serde_json::to_string(metadata)?),
    ] {
        string(&mut out, k);
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, v);
    }
    let mut offset = 0;
    for tensor in tensors {
        string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in tensor.shape.iter().rev() {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.gguf_type().to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        offset += align(tensor.bytes.len());
    }
    out.resize(align(out.len()), 0);
    for tensor in tensors {
        out.extend_from_slice(tensor.bytes);
        out.resize(align(out.len()), 0);
    }
    Ok(out)
}

fn take<'a>(r: &mut Cursor<&'a [u8]>, len: usize) -> Result<&'a [u8]> {
    let i = usize::try_from(r.position())?;
    let end = i.checked_add(len).context("GGUF field overflow")?;
    let value = r.get_ref().get(i..end).context("GGUF is truncated")?;
    r.set_position(end as u64);
    Ok(value)
}

fn gguf_string<'a>(r: &mut Cursor<&'a [u8]>, max: usize) -> Result<&'a str> {
    let len = usize::try_from(u64le(take(r, 8)?, 0))?;
    ensure!(len <= max, "GGUF string exceeds limit");
    Ok(std::str::from_utf8(take(r, len)?)?)
}

fn read_gguf<'a>(
    bytes: &'a [u8],
    expected: &[Tensor<'_>; 3],
    manifest: &Value,
) -> Result<[&'a [u8]; 3]> {
    let mut r = Cursor::new(bytes);
    ensure!(
        take(&mut r, 8)? == b"GGUF\x03\0\0\0"
            && u64le(take(&mut r, 8)?, 0) == 3
            && u64le(take(&mut r, 8)?, 0) == 2,
        "unsupported GGUF header"
    );
    for (k, v) in [
        ("general.architecture", "brewery-hgn"),
        (SCHEMA, &serde_json::to_string(manifest)?),
    ] {
        ensure!(
            gguf_string(&mut r, 128)? == k
                && u32le(take(&mut r, 4)?, 0) == 8
                && gguf_string(&mut r, 8192)? == v,
            "GGUF metadata differs from source binding"
        );
    }
    let mut start = 0;
    for t in expected {
        ensure!(
            gguf_string(&mut r, 96)? == t.name
                && u32le(take(&mut r, 4)?, 0) as usize == t.shape.len(),
            "GGUF tensor name or rank mismatch"
        );
        for dim in t.shape.iter().rev() {
            ensure!(u64le(take(&mut r, 8)?, 0) == *dim, "GGUF dimensions differ");
        }
        ensure!(
            u32le(take(&mut r, 4)?, 0) == t.gguf_type()
                && u64le(take(&mut r, 8)?, 0) == start as u64,
            "GGUF type or offset differs"
        );
        start += align(t.bytes.len());
    }
    let pos = r.position() as usize;
    ensure!(
        take(&mut r, align(pos) - pos)?.iter().all(|b| *b == 0),
        "nonzero GGUF header padding"
    );
    let mut parts: [&[u8]; 3] = [&[]; 3];
    for (t, part) in expected.iter().zip(&mut parts) {
        *part = take(&mut r, t.bytes.len())?;
        ensure!(
            *part == t.bytes,
            "GGUF encoded payload or companion differs"
        );
        ensure!(
            take(&mut r, align(part.len()) - part.len())?
                .iter()
                .all(|b| *b == 0),
            "nonzero GGUF tensor padding"
        );
    }
    ensure!(r.position() == bytes.len() as u64, "trailing GGUF bytes");
    Ok(parts)
}

fn proof(source: &[u8], t: &Tensor<'_>, payloads: [&[u8]; 3]) -> Result<(String, String)> {
    let enc = t.encoding().unwrap();
    let (experts, rows, cols) = geometry(&t.shape, enc)?;
    let name = t
        .name
        .strip_prefix("layers.")
        .context("v1 native oracle requires a layers.* tensor")?;
    let name = if enc == Encoding::I4r {
        ensure!(
            name.ends_with(".mlp.experts.down_proj.weight"),
            "v1 I4R supports routed down projections only"
        );
        name.strip_suffix(".weight").unwrap()
    } else {
        ensure!(
            !name.contains(".mlp.experts."),
            "v1 HT supports rank-two non-expert weights only"
        );
        name
    };
    let canonical = format!("model.language_model.layers.{name}");
    // Oracle uses the exact bounded original file, including its original
    // tensor table and offsets, isolated from concurrent caller mutations.
    let mut snapshot = tempfile::NamedTempFile::new()?;
    snapshot.write_all(source)?;
    let oracle = HgnSidecar::open(snapshot.path())?;
    oracle.hash_tensor(&canonical)?;
    let vector = |raw: &[u8]| {
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect::<Vec<_>>()
    };
    let su = vector(payloads[1]);
    let sv = vector(payloads[2]);
    let band = 128 * cols;
    let mut decoded = Sha256::new();
    let mut product = Sha256::new();
    for index in (0..experts * rows).step_by(128) {
        let start = index * cols;
        let block = if enc == Encoding::I4r {
            let layout = codec::payload_layout(enc, experts * rows, cols)?;
            let scale = layout.scales_offset as usize + index * layout.scale_row_stride as usize;
            EncodedBlock {
                codes: payloads[0][start / 2..(start + band) / 2].into(),
                scales: payloads[0][scale..scale + 128 * layout.scale_row_stride as usize].into(),
            }
        } else {
            let bits = enc.ht_bits().unwrap();
            EncodedBlock {
                codes: payloads[0][start * bits / 8..(start + band) * bits / 8].into(),
                scales: vec![],
            }
        };
        let local = index % rows;
        let result = codec::decode_rows(
            enc,
            &block,
            128,
            cols,
            &QuantMeta {
                su: &su,
                sv: &sv[local..local + 128],
                ..Default::default()
            },
        )?;
        let reference = oracle.read_f32(&canonical, start as u64, band)?;
        ensure!(
            result
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "native HGN decode mismatch"
        );
        for value in &result {
            decoded.update(value.to_le_bytes());
        }
        let dot = |row: &[f32]| {
            row.iter()
                .enumerate()
                .map(|(i, &v)| f64::from(v) * ((i % 23) as f64 - 11.0) / 16.0)
                .sum::<f64>()
        };
        for (a, b) in result.chunks_exact(cols).zip(reference.chunks_exact(cols)) {
            let value = dot(a);
            ensure!(
                value.is_finite() && value.to_bits() == dot(b).to_bits(),
                "CPU matvec mismatch"
            );
            product.update(value.to_le_bytes());
        }
    }
    Ok((
        format!("{:x}", decoded.finalize()),
        format!("{:x}", product.finalize()),
    ))
}

fn run() -> Result<Value> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    ensure!(
        (args.len() == 3 || args.len() == 5 && args[3] == "--tensor")
            && matches!(args[0].to_str(), Some("pack" | "verify")),
        "usage: pack|verify SOURCE DEST [--tensor NAME]"
    );
    let source_path = Path::new(&args[1]);
    let dest = Path::new(&args[2]);
    let requested = args
        .get(4)
        .map(|s| s.to_str().context("tensor name must be UTF-8"))
        .transpose()?;
    let source = read(source_path)?;
    let (tensors, mut manifest) = selection(&source, requested)?;
    let packing = args[0] == "pack";
    let bytes = if packing {
        gguf(&tensors, &manifest)?
    } else {
        read(dest)?
    };
    let parts = read_gguf(&bytes, &tensors, &manifest)?;
    let (decoded, product) = proof(&source, &tensors[0], parts)?;
    ensure!(
        read(source_path)? == source,
        "HGN source changed during proof"
    );
    if packing {
        let parent = dest
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(&bytes)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(dest)?;
    } else {
        ensure!(read(dest)? == bytes, "GGUF artifact changed during proof");
    }
    manifest["gguf_sha256"] = json!(hash(&bytes));
    manifest["gguf_bytes"] = json!(bytes.len());
    manifest["native_decode_bitwise_equal"] = json!(true);
    manifest["cpu_matvec_bitwise_equal"] = json!(true);
    manifest["decoded_f32_sha256"] = json!(decoded);
    manifest["matvec_f64_sha256"] = json!(product);
    manifest["release_qualified"] = json!(false);
    Ok(manifest)
}

fn main() {
    match run() {
        Ok(receipt) => println!("{receipt}"),
        Err(error) => {
            eprintln!("hgn-in-gguf: {error:#}");
            std::process::exit(1);
        }
    }
}
