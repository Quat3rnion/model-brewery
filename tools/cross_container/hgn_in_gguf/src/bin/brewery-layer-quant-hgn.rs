//! Bounded CPU adapter for the layer-recipe frontend. The upstream library
//! owns the native encoding, tensor table, alignment and checksums.
use anyhow::{bail, ensure, Context, Result};
use bf16_to_hgn::{
    codec::{encode_rows, generate_signs, ht_row_scales, Encoding, QuantMeta},
    container::{Reader, Writer},
    rows_cols, ContainerPlan, Origin, Rotation, TensorPlan, ENCODER_SHA256, SPEC_COMMIT,
};
use half::{bf16, f16};
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{self, BufRead, Read},
    path::Path,
};

const MAX_PLAN: usize = 2 << 20;
const MAX_SOURCE: usize = 64 << 20;
const MAX_HGN: u64 = 129 << 20;
const IDENTITY: &str = "brewery-layer-recipe-structural-v1";
const ROTATION_SEED: u64 = 0; // Stock converter default.

fn encoding(label: &str) -> Result<Encoding> {
    match label {
        "BF16" => Ok(Encoding::Bf16),
        "F16" => Ok(Encoding::F16),
        "F32" => Ok(Encoding::F32),
        "Q8G64" => Ok(Encoding::Q8G64),
        "HT3" => Ok(Encoding::Ht3),
        "HT4" => Ok(Encoding::Ht),
        "Q8_0" => bail!("GGML Q8_0 has no native HGN representation; Q8G64 is a different codec"),
        _ => bail!("unsupported native HGN codec: {label}"),
    }
}

fn label(encoding: Encoding) -> Result<&'static str> {
    match encoding {
        Encoding::Bf16 => Ok("BF16"),
        Encoding::F16 => Ok("F16"),
        Encoding::F32 => Ok("F32"),
        Encoding::Q8G64 => Ok("Q8G64"),
        Encoding::Ht3 => Ok("HT3"),
        Encoding::Ht => Ok("HT4"),
        _ => bail!("unsupported native codec for this bounded HGN adapter"),
    }
}

fn width(dtype: Encoding) -> Result<usize> {
    match dtype {
        Encoding::Bf16 | Encoding::F16 => Ok(2),
        Encoding::F32 => Ok(4),
        _ => bail!("SafeTensors input requires BF16, F16, or F32"),
    }
}

fn rotation_names(name: &str) -> Result<Rotation> {
    let base = name
        .strip_suffix(".weight")
        .context("native HT requires a .weight tensor name")?;
    Ok(Rotation {
        su: format!("{base}.suh"),
        sv: format!("{base}.svh"),
    })
}

// Keep the upstream converter's Rust 1.85 minimum (as_chunks stabilized in 1.88).
#[allow(clippy::chunks_exact_to_as_chunks)]
fn floats(raw: &[u8], dtype: Encoding) -> Result<Vec<f32>> {
    let values: Vec<_> = match dtype {
        Encoding::Bf16 => raw
            .chunks_exact(2)
            .map(|p| bf16::from_le_bytes([p[0], p[1]]).to_f32())
            .collect(),
        Encoding::F16 => raw
            .chunks_exact(2)
            .map(|p| f16::from_le_bytes([p[0], p[1]]).to_f32())
            .collect(),
        Encoding::F32 => raw
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect(),
        _ => bail!("unsupported source float"),
    };
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite source value cannot change dtype"
    );
    Ok(values)
}

fn encode(
    to: Encoding,
    values: &[f32],
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<Vec<u8>> {
    let result = encode_rows(to, values, rows, cols, meta)?;
    ensure!(
        result.scales.is_empty(),
        "unexpected detached native scale payload"
    );
    Ok(result.codes)
}

fn write_payload(writer: &mut Writer, index: usize, payload: &[u8]) -> Result<()> {
    let entry = &writer.layout.entries[index];
    ensure!(
        payload.len() as u64 == entry.size,
        "native size mismatch: {}",
        entry.name
    );
    writer.write_at(index, 0, payload)
}

fn write_tensor(
    writer: &mut Writer,
    index: usize,
    tensor: &TensorPlan,
    raw: Vec<u8>,
    from: Encoding,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if from == tensor.encoding {
        return write_payload(writer, index, &raw);
    }
    let values = floats(&raw, from)?;
    if let Some(rotation) = &tensor.rotation {
        let su = generate_signs(ROTATION_SEED, &rotation.su, cols);
        let signs = generate_signs(ROTATION_SEED, &rotation.sv, rows);
        let sv = ht_row_scales(&values, rows, cols, &signs)?;
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ..QuantMeta::default()
        };
        write_payload(
            writer,
            index,
            &encode(tensor.encoding, &values, rows, cols, &meta)?,
        )?;
        for (i, plane) in [su, sv].into_iter().enumerate() {
            let len = plane.len();
            write_payload(
                writer,
                index + 1 + i,
                &encode(Encoding::F16, &plane, 1, len, &QuantMeta::default())?,
            )?;
        }
    } else {
        write_payload(
            writer,
            index,
            &encode(tensor.encoding, &values, rows, cols, &QuantMeta::default())?,
        )?;
    }
    Ok(())
}

fn write(path: &Path, mut input: impl BufRead) -> Result<()> {
    let mut header = Vec::new();
    input
        .by_ref()
        .take(MAX_PLAN as u64)
        .read_until(b'\n', &mut header)?;
    ensure!(
        header.last() == Some(&b'\n'),
        "missing or oversized native HGN adapter plan"
    );
    let entries: Vec<Value> =
        serde_json::from_slice(&header).context("invalid HGN adapter plan")?;
    ensure!(
        (1..=4096).contains(&entries.len()),
        "invalid HGN adapter tensor count"
    );
    let mut plan = ContainerPlan {
        filename: "model.hgn".into(),
        identity: IDENTITY.into(),
        tensors: Vec::with_capacity(entries.len()),
    };
    let mut sources = Vec::with_capacity(entries.len());
    let mut total = 0usize;
    for item in entries {
        let obj = item
            .as_object()
            .context("expected tensor descriptor object")?;
        ensure!(obj.len() == 4, "invalid tensor descriptor fields");
        let name = obj["name"]
            .as_str()
            .context("invalid tensor name")?
            .to_owned();
        let shape: Vec<u64> =
            serde_json::from_value(obj["shape"].clone()).context("invalid shape")?;
        let (rows, cols) = rows_cols(&shape).with_context(|| format!("invalid shape: {name}"))?;
        let from = encoding(obj["dtype"].as_str().context("invalid source dtype")?)?;
        let to = encoding(obj["target"].as_str().context("invalid target dtype")?)?;
        ensure!(
            to != Encoding::Q8G64 || shape.len() == 2 && cols % 64 == 0,
            "native Q8G64 requires a matrix with columns divisible by 64: {name}"
        );
        let rotation = if to.ht_bits().is_some() {
            ensure!(
                shape.len() == 2
                    && shape
                        .iter()
                        .all(|d| (128..=1 << 24).contains(d) && d % 128 == 0),
                "{} requires rows and columns divisible by 128: {name}",
                label(to)?
            );
            Some(rotation_names(&name)?)
        } else {
            None
        };
        let source_width = width(from)?;
        ensure!(
            to == Encoding::Q8G64 || rotation.is_some() || width(to)? <= source_width,
            "widening is not supported: {name}"
        );
        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(source_width))
            .context("oversized source tensor")?;
        total = total.checked_add(bytes).context("source size overflow")?;
        ensure!(total <= MAX_SOURCE, "native HGN fixture exceeds 64 MiB");
        let index = plan.tensors.len();
        plan.tensors.push(TensorPlan {
            name: name.clone(),
            shape,
            encoding: to,
            origin: Origin::Tensor { name: name.clone() },
            rotation: rotation.clone(),
        });
        if let Some(Rotation { su, sv }) = rotation {
            plan.tensors.extend([
                TensorPlan {
                    name: su.clone(),
                    shape: vec![cols as u64],
                    encoding: Encoding::F16,
                    origin: Origin::Signs { key: su },
                    rotation: None,
                },
                TensorPlan {
                    name: sv.clone(),
                    shape: vec![rows as u64],
                    encoding: Encoding::F16,
                    origin: Origin::HtScales {
                        tensor: name,
                        signs_key: sv,
                    },
                    rotation: None,
                },
            ]);
        }
        sources.push((index, from, rows, cols, bytes));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staging = tempfile::tempdir_in(parent)?;
    let staged = staging.path().join("model.hgn");
    let mut writer = Writer::create(&staged, &plan)?;
    for (index, from, rows, cols, bytes) in sources {
        let tensor = &plan.tensors[index];
        let mut raw = vec![0; bytes];
        input
            .read_exact(&mut raw)
            .with_context(|| format!("truncated payload: {}", tensor.name))?;
        write_tensor(&mut writer, index, tensor, raw, from, rows, cols)
            .with_context(|| format!("cannot encode {}", tensor.name))?;
    }
    ensure!(
        input.fill_buf()?.is_empty(),
        "trailing source payload bytes"
    );
    writer.finish()?;
    fs::hard_link(staged, path).context("publish new HGN without replacing destination")?;
    Ok(())
}

fn inspect(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_HGN,
        "expected a bounded HGN fixture file"
    );
    let mut reader = Reader::open(path)?;
    ensure!(reader.version == 2, "expected native HGN version 2");
    let mut inventory = Vec::with_capacity(reader.layout.entries.len());
    for index in 0..reader.layout.entries.len() {
        let fingerprint = reader.fingerprint(index)?;
        let entry = &reader.layout.entries[index];
        ensure!(
            entry.checksum == fingerprint.xor32,
            "native HGN checksum mismatch: {}",
            entry.name
        );
        inventory.push(json!({
            "name": entry.name,
            "shape": entry.shape,
            "dtype": label(entry.encoding)?,
            "code": entry.encoding.code(),
            "variant": entry.encoding.variant(),
            "offset": entry.offset,
            "size": entry.size,
            "sha256": fingerprint.sha256,
        }));
    }
    println!(
        "{}",
        json!({
            "version": reader.version,
            "identity": reader.layout.identity,
            "encoder_sha256": ENCODER_SHA256,
            "spec_commit": SPEC_COMMIT,
            "tensors": inventory,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "write" => write(Path::new(path), io::stdin().lock()),
        [cmd, path] if cmd == "inspect" => inspect(Path::new(path)),
        _ => bail!("usage: brewery-layer-quant-hgn write|inspect PATH"),
    }
}
