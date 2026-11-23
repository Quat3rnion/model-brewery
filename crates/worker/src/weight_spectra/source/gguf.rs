use super::*;
use std::io::BufReader;

type Reader<'a> = BufReader<&'a mut File>;

pub(super) fn inventory(file: &mut File, bytes: u64) -> Result<BTreeMap<String, Entry>> {
    let mut r = BufReader::new(file);
    let mut magic = [0; 4];
    r.read_exact(&mut magic)?;
    ensure!(
        &magic == b"GGUF" && matches!(uint::<4>(&mut r)?, 2 | 3),
        "Expected a GGUF V2 or V3 file"
    );
    let count = uint::<8>(&mut r)?;
    let kvs = uint::<8>(&mut r)?;
    let limit = bytes.min(128 << 20);
    ensure!(
        (1..=MAX_TENSORS as u64).contains(&count) && kvs <= 8192,
        "GGUF inventory exceeds limits"
    );
    let mut alignment = 32;
    let mut unique = BTreeSet::new();
    for _ in 0..kvs {
        let key = text(&mut r, limit, 4096)?;
        ensure!(unique.insert(key.clone()), "Duplicate GGUF metadata key");
        let dtype = uint::<4>(&mut r)?;
        match key.as_str() {
            "general.alignment" => {
                ensure!(dtype == 4, "Invalid GGUF alignment type");
                alignment = uint::<4>(&mut r)?;
                ensure!(
                    alignment.is_power_of_two() && alignment <= 4096,
                    "Invalid GGUF alignment"
                );
            }
            "split.count" | "split.no" => {
                ensure!(
                    matches!(dtype, 0 | 2 | 4 | 10),
                    "Invalid GGUF split metadata"
                );
                let width = scalar(dtype)? as usize;
                let mut encoded = [0; 8];
                r.read_exact(&mut encoded[..width])?;
                ensure!(
                    u64::from_le_bytes(encoded) == u64::from(key == "split.count"),
                    "Split GGUF is not supported"
                );
            }
            _ if dtype == 9 => {
                let member = uint::<4>(&mut r)?;
                let length = uint::<8>(&mut r)?;
                if member == 8 {
                    ensure!(length <= 2_000_000, "GGUF string array too large");
                    for _ in 0..length {
                        let n = uint::<8>(&mut r)?;
                        skip(&mut r, limit, n)?;
                    }
                } else {
                    skip(
                        &mut r,
                        limit,
                        length
                            .checked_mul(scalar(member)?)
                            .context("GGUF array overflow")?,
                    )?;
                }
            }
            _ => {
                let width = if dtype == 8 {
                    uint::<8>(&mut r)?
                } else {
                    scalar(dtype)?
                };
                skip(&mut r, limit, width)?;
            }
        }
    }
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        let name = text(&mut r, limit, 95)?;
        ensure!(
            crate::weight_source::native_tensor_name(&name),
            "Invalid GGUF tensor name"
        );
        let rank = uint::<4>(&mut r)?;
        ensure!(
            (1..=4).contains(&rank),
            "Invalid GGUF tensor dimension count"
        );
        let mut shape = (0..rank)
            .map(|_| uint::<8>(&mut r))
            .collect::<Result<Vec<_>>>()?;
        ensure!(shape.iter().all(|&v| v > 0), "Invalid GGUF tensor shape");
        shape.reverse();
        let code = uint::<4>(&mut r)?;
        let dtype = match code {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            8 => "Q8_0",
            12 => "Q4_K",
            14 => "Q6_K",
            30 => "BF16",
            _ => "",
        };
        let offset = uint::<8>(&mut r)?;
        ensure!(offset % alignment == 0, "Unaligned GGUF tensor offset");
        let key = canonical_name(&name);
        insert(
            &mut entries,
            key,
            Entry {
                name,
                shape,
                offset,
                file: 0,
                dtype: if dtype.is_empty() {
                    format!("GGML_TYPE_{code}")
                } else {
                    dtype.into()
                },
            },
        )?;
    }
    let end = r.stream_position()?;
    ensure!(end <= limit, "GGUF header exceeds limits");
    let base = (end + alignment - 1) & !(alignment - 1);
    let mut offsets: Vec<_> = entries.values_mut().collect();
    offsets.sort_unstable_by_key(|e| e.offset);
    let mut right = bytes;
    for t in offsets.into_iter().rev() {
        t.offset = base
            .checked_add(t.offset)
            .context("GGUF data offset overflow")?;
        ensure!(t.offset < right, "Invalid GGUF tensor offset");
        if let Some((block, item)) = quant_layout(&t.dtype) {
            ensure!(
                t.shape.last().unwrap() % block as u64 == 0,
                "Invalid GGUF block width"
            );
            let size = t
                .shape
                .iter()
                .try_fold(1u64, |n, x| n.checked_mul(*x))
                .and_then(|n| (n / block as u64).checked_mul(item as u64))
                .context("GGUF tensor size overflow")?;
            ensure!(
                size <= right - t.offset,
                "Truncated or overlapping GGUF tensor"
            );
        }
        right = t.offset;
    }
    Ok(entries)
}

fn canonical_name(name: &str) -> String {
    for (src, dst) in [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
    ] {
        if name == src {
            return dst.into();
        }
    }
    if let Some((layer, rest)) = name.strip_prefix("blk.").and_then(|s| s.split_once('.'))
        && layer.parse::<u32>().is_ok()
        && let Some((role, param @ ("weight" | "bias"))) = rest.split_once('.')
    {
        let mapped = match role {
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_output" => "self_attn.o_proj",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            "ffn_down" => "mlp.down_proj",
            "attn_norm" => "input_layernorm",
            "ffn_norm" => "post_attention_layernorm",
            _ => return name.into(),
        };
        return format!("model.layers.{layer}.{mapped}.{param}");
    }
    name.into()
}

pub(super) fn dequantize(dtype: &str, b: &[u8], out: &mut Vec<f64>) {
    // GGML block layouts and f32 arithmetic:
    // https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
    let half = |i| decode("F16", &b[i..i + 2]) as f32;
    match dtype {
        "Q4_0" => {
            let d = half(0);
            for shift in [0, 4] {
                out.extend(
                    b[2..]
                        .iter()
                        .map(|&q| f64::from(d * f32::from(((q >> shift) & 15) as i8 - 8))),
                );
            }
        }
        "Q8_0" => {
            let d = half(0);
            out.extend(b[2..].iter().map(|&q| f64::from(d * f32::from(q as i8))));
        }
        "Q4_K" => {
            let (d, m) = (half(0), half(2));
            let scales = &b[4..16];
            for j in 0..8 {
                let (sc, mn) = if j < 4 {
                    (scales[j] & 63, scales[j + 4] & 63)
                } else {
                    (
                        (scales[j + 4] & 15) | (scales[j - 4] >> 6) << 4,
                        (scales[j + 4] >> 4) | (scales[j] >> 6) << 4,
                    )
                };
                let start = 16 + (j / 2) * 32;
                let shift = j % 2 * 4;
                out.extend(b[start..start + 32].iter().map(|&q| {
                    f64::from(
                        (d * f32::from(sc)) * f32::from((q >> shift) & 15) - m * f32::from(mn),
                    )
                }));
            }
        }
        "Q6_K" => {
            let d = half(208);
            for half_block in 0..2 {
                for j in 0..4 {
                    for i in 0..32 {
                        let low = b[half_block * 64 + (j % 2) * 32 + i] >> (j / 2 * 4) & 15;
                        let high = b[128 + half_block * 32 + i] >> (j * 2) & 3;
                        let q = ((high << 4) | low) as i8 - 32;
                        let sc = b[192 + half_block * 8 + j * 2 + i / 16] as i8;
                        out.push(f64::from(d * f32::from(sc) * f32::from(q)));
                    }
                }
            }
        }
        _ => unreachable!("selected quantized layout"),
    }
}

fn uint<const N: usize>(r: &mut Reader<'_>) -> Result<u64> {
    let mut buf = [0; 8];
    r.read_exact(&mut buf[..N])?;
    Ok(u64::from_le_bytes(buf))
}

fn scalar(t: u64) -> Result<u64> {
    Ok(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => anyhow::bail!("Unsupported GGUF metadata type"),
    })
}

fn skip(r: &mut Reader<'_>, limit: u64, n: u64) -> Result<()> {
    ensure!(
        n <= limit.saturating_sub(r.stream_position()?),
        "GGUF metadata exceeds limits"
    );
    r.seek_relative(n as i64)?;
    Ok(())
}

fn text(r: &mut Reader<'_>, limit: u64, max: u64) -> Result<String> {
    let n = uint::<8>(r)?;
    ensure!(
        n <= max.min(limit.saturating_sub(r.stream_position()?)),
        "GGUF string exceeds limits"
    );
    let mut data = vec![0; n as usize];
    r.read_exact(&mut data)?;
    Ok(String::from_utf8(data)?)
}
