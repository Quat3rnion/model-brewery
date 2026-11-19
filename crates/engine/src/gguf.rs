//! Check metadata that affects the set of files and the trace engine selected.
//! The native reader still validates tensor metadata and loads the model.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
};

pub fn trace_metadata(path: &Path, attested_shards: usize) -> Result<Value> {
    let file = File::open(path)?;
    let limit = file.metadata()?.len().min(128 << 20);
    let mut r = BufReader::new(file);
    let mut magic = [0; 4];
    r.read_exact(&mut magic)?;
    ensure!(
        &magic == b"GGUF" && matches!(number(&mut r, 4)?, 2 | 3),
        "unsupported GGUF header"
    );
    ensure!(number(&mut r, 8)? > 0, "GGUF contains no tensors");
    let count = number(&mut r, 8)?;
    ensure!(count < 65536, "too many GGUF metadata fields");
    let (mut arch, mut variant, mut template, mut split_count, mut split_index) =
        (None, None, None, None, None);
    for _ in 0..count {
        let key = read_string(&mut r, limit, 4096)?;
        let ty = number(&mut r, 4)?;
        match key.as_str() {
            "general.architecture" | "deepseek4.variant" | "tokenizer.chat_template" => {
                let dst = match key.as_str() {
                    "general.architecture" => &mut arch,
                    "deepseek4.variant" => &mut variant,
                    _ => &mut template,
                };
                ensure!(
                    ty == 8 && dst.is_none(),
                    "invalid or duplicate {key} metadata"
                );
                *dst = Some(read_string(&mut r, limit, 256 << 10)?);
            }
            "split.count" | "split.no" => {
                let dst = if key == "split.count" {
                    &mut split_count
                } else {
                    &mut split_index
                };
                let width = match ty {
                    0 => 1,
                    2 => 2,
                    4 => 4,
                    10 => 8,
                    _ => bail!("invalid {key} type"),
                };
                ensure!(dst.is_none(), "duplicate {key}");
                *dst = Some(number(&mut r, width)?);
            }
            _ if ty == 9 => {
                let member = number(&mut r, 4)?;
                let size = number(&mut r, 8)?;
                if member == 8 {
                    ensure!(size <= 2_000_000, "GGUF string array too large");
                    for _ in 0..size {
                        let len = number(&mut r, 8)?;
                        skip(&mut r, limit, len)?;
                    }
                } else {
                    let bytes = size
                        .checked_mul(scalar_width(member)?)
                        .context("GGUF array size overflow")?;
                    skip(&mut r, limit, bytes)?;
                }
            }
            _ => {
                let len = if ty == 8 {
                    number(&mut r, 8)?
                } else {
                    scalar_width(ty)?
                };
                skip(&mut r, limit, len)?;
            }
        }
        ensure!(
            r.stream_position()? <= limit,
            "GGUF metadata exceeds inspection bounds"
        );
    }
    ensure!(
        arch.as_deref().is_some_and(|a| !a.is_empty()),
        "GGUF architecture missing"
    );
    ensure!(
        arch.as_deref() != Some("deepseek4") || variant.as_deref() != Some("v41_flash"),
        "native routing captures are not supported by GGUF compare"
    );
    ensure!(
        split_count.unwrap_or(1).max(1) == attested_shards as u64 && split_index.unwrap_or(0) == 0,
        "GGUF split metadata disagrees with staged shards"
    );
    let template_sha = template
        .as_ref()
        .map(|s| format!("{:x}", Sha256::digest(s.as_bytes())));
    Ok(
        json!({"architecture":arch,"embedded_default_template_sha256":template_sha,
              "staged_shards":attested_shards}),
    )
}

pub(crate) fn scalar_width(ty: u64) -> Result<u64> {
    Ok(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => bail!("unsupported GGUF metadata type {ty}"),
    })
}

pub(crate) fn number(r: &mut impl Read, width: usize) -> Result<u64> {
    let mut bytes = [0; 8];
    r.read_exact(&mut bytes[..width])?;
    Ok(u64::from_le_bytes(bytes))
}

pub(crate) fn skip(r: &mut impl Seek, limit: u64, n: u64) -> Result<()> {
    let end = r
        .stream_position()?
        .checked_add(n)
        .context("GGUF metadata offset overflow")?;
    ensure!(end <= limit, "GGUF metadata exceeds inspection bounds");
    r.seek_relative(n.try_into()?)?;
    Ok(())
}

pub(crate) fn read_string(r: &mut (impl Read + Seek), limit: u64, max: u64) -> Result<String> {
    let len = number(r, 8)?;
    ensure!(
        len <= max && len <= limit.saturating_sub(r.stream_position()?),
        "GGUF metadata string exceeds bounds"
    );
    let mut bytes = vec![0; usize::try_from(len)?];
    r.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}
