use crate::gguf::{number, read_string, scalar_width, skip};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
};

pub(super) type Types = BTreeMap<String, u64>;
const TYPES: &[(u64, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
    (40, "NVFP4", 64, 36),
    (41, "Q1_0", 128, 18),
    (42, "Q2_0", 64, 18),
];

pub(super) fn exact_dtype(name: &str) -> Option<u64> {
    TYPES
        .iter()
        .find(|(id, n, _, _)| *n == name && !matches!(id, 0 | 24..=27))
        .map(|entry| entry.0)
}

pub(super) fn dtype(name: &str) -> Option<u64> {
    TYPES
        .iter()
        .find(|(_, n, _, _)| n.eq_ignore_ascii_case(name))
        .map(|entry| entry.0)
}

fn spec(id: u64) -> Result<&'static (u64, &'static str, u64, u64)> {
    TYPES
        .iter()
        .find(|entry| entry.0 == id)
        .with_context(|| format!("unsupported GGUF tensor type {id}"))
}

pub(super) fn valid_name(name: &str) -> bool {
    (1..=63).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

pub(super) struct Tensor {
    pub(super) shape: Vec<u64>,
    pub(super) ty: u64,
}

impl Tensor {
    pub(super) fn bytes(&self, ty: u64) -> Result<u64> {
        let &(_, _, block, size) = spec(ty)?;
        ensure!(
            self.shape[0].is_multiple_of(block),
            "GGUF tensor row is incompatible with requested block size"
        );
        self.shape[1..].iter().try_fold(
            (self.shape[0] / block)
                .checked_mul(size)
                .context("GGUF size overflow")?,
            |acc, n| acc.checked_mul(*n).context("GGUF size overflow"),
        )
    }
}

pub(super) struct Model {
    pub(super) architecture: String,
    pub(super) tensors: BTreeMap<String, Tensor>,
    ftype: Option<u64>,
    align: u64,
    data: u64,
}

fn align(value: u64, align: u64) -> Result<u64> {
    Ok(value
        .checked_add(align - 1)
        .context("GGUF alignment overflow")?
        & !(align - 1))
}

impl Model {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let meta = crate::integrity::canonical(path)?;
        ensure!(meta.is_file(), "expected regular GGUF file");
        let mut r = BufReader::new(File::open(path)?);
        let limit = meta.len().min(128 << 20);
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;
        ensure!(
            &magic == b"GGUF" && matches!(number(&mut r, 4)?, 2 | 3),
            "unsupported GGUF header"
        );
        let nt = number(&mut r, 8)?;
        let nk = number(&mut r, 8)?;
        ensure!(
            (1..=32768).contains(&nt) && nk <= 65535,
            "GGUF inventory exceeds bounds"
        );
        let mut fields = BTreeSet::new();
        let (mut architecture, mut ftype, mut alignment) = (None, None, 32);
        for _ in 0..nk {
            let key = read_string(&mut r, limit, 4096)?;
            ensure!(fields.insert(key.clone()), "duplicate GGUF metadata key");
            let ty = number(&mut r, 4)?;
            match key.as_str() {
                "general.architecture" => {
                    ensure!(ty == 8, "invalid GGUF architecture");
                    architecture = Some(read_string(&mut r, limit, 128)?);
                }
                "general.file_type" | "general.alignment" => {
                    ensure!(ty == 4, "invalid GGUF integer metadata");
                    let v = number(&mut r, 4)?;
                    if key == "general.file_type" {
                        ftype = Some(v);
                    } else {
                        ensure!(
                            (1..=4096).contains(&v) && v.is_power_of_two(),
                            "invalid GGUF alignment"
                        );
                        alignment = v;
                    }
                }
                "split.count" | "split.no" | "split.tensors.count" => {
                    bail!("split GGUF is not supported")
                }
                _ if ty == 9 => {
                    let ty = number(&mut r, 4)?;
                    let n = number(&mut r, 8)?;
                    if ty == 8 {
                        ensure!(n <= 2_000_000, "GGUF string array exceeds bounds");
                        for _ in 0..n {
                            let len = number(&mut r, 8)?;
                            skip(&mut r, limit, len)?;
                        }
                    } else {
                        let len = n
                            .checked_mul(scalar_width(ty)?)
                            .context("GGUF array size overflow")?;
                        skip(&mut r, limit, len)?;
                    }
                }
                _ => {
                    let n = if ty == 8 {
                        number(&mut r, 8)?
                    } else {
                        scalar_width(ty)?
                    };
                    skip(&mut r, limit, n)?;
                }
            }
        }
        let architecture = architecture
            .filter(|name| valid_name(name))
            .context("missing/invalid GGUF architecture")?;
        let mut spans = Vec::new();
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = read_string(&mut r, limit, 63)?;
            ensure!(valid_name(&name), "invalid GGUF tensor name");
            let rank = number(&mut r, 4)?;
            ensure!((1..=4).contains(&rank), "invalid GGUF tensor rank");
            let mut shape = Vec::new();
            for _ in 0..rank {
                let dim = number(&mut r, 8)?;
                ensure!(
                    (1..=i64::MAX as u64).contains(&dim),
                    "invalid GGUF tensor dimension"
                );
                shape.push(dim);
            }
            let ty = number(&mut r, 4)?;
            let offset = number(&mut r, 8)?;
            let tensor = Tensor { shape, ty };
            let end = offset
                .checked_add(tensor.bytes(ty)?)
                .context("GGUF tensor extent overflow")?;
            ensure!(
                offset % alignment == 0 && end <= meta.len(),
                "GGUF tensor offset or size exceeds bounds"
            );
            ensure!(
                tensors.insert(name, tensor).is_none(),
                "duplicate GGUF tensor name"
            );
            spans.push((offset, end));
        }
        let data = align(r.stream_position()?, alignment)?;
        ensure!(
            data <= limit && data <= meta.len(),
            "GGUF tensor table exceeds bounds"
        );
        spans.sort_unstable();
        let mut end = 0;
        for (start, next) in spans {
            ensure!(
                start >= end && next <= meta.len() - data,
                "GGUF tensor payload overlaps or is truncated"
            );
            end = next;
        }
        Ok(Self {
            architecture,
            tensors,
            ftype,
            align: alignment,
            data,
        })
    }

    pub(super) fn predicted_bytes(&self, types: &Types) -> Result<u64> {
        self.tensors.iter().try_fold(self.data, |n, (name, t)| {
            n.checked_add(align(t.bytes(types[name])?, self.align)?)
                .context("GGUF size estimate overflow")
        })
    }

    pub(super) fn summary(&self, types: &Types) -> Value {
        json!(self
            .tensors
            .iter()
            .map(|(name, t)| json!({
                "name":name,"shape":t.shape,"current_type":spec(t.ty).unwrap().1,
                "planned_type":types.get(name).and_then(|id|spec(*id).ok()).map(|s|s.1)
            }))
            .collect::<Vec<_>>())
    }

    pub(super) fn parse_plan(
        &self,
        log: &str,
        overrides: &BTreeMap<String, String>,
    ) -> Result<Types> {
        let mut result = BTreeMap::new();
        for line in log.lines() {
            let Some((number, tail)) = line.strip_prefix('[').and_then(|l| l.split_once(']'))
            else {
                continue;
            };
            let Some((index, total)) = number.split_once('/') else {
                continue;
            };
            if index.trim().parse::<usize>().is_err() || total.trim().parse::<usize>().is_err() {
                continue;
            }
            ensure!(
                total.trim().parse::<usize>()? == self.tensors.len(),
                "native GGUF plan tensor count differs"
            );
            let (name, info) = tail
                .split_once(" - [")
                .context("invalid native GGUF plan line")?;
            let name = name.trim();
            let tensor = self
                .tensors
                .get(name)
                .context("native GGUF plan contains unknown tensor")?;
            let (_, ty_line) = info
                .split_once("], type = ")
                .context("missing native GGUF plan type")?;
            let (current, size) = ty_line
                .split_once(", size = ")
                .context("missing native GGUF plan size")?;
            ensure!(
                dtype(current.trim()) == Some(tensor.ty),
                "native GGUF plan input type differs"
            );
            let ty = if size.contains(" -> ") {
                let (_, ty) = size
                    .rsplit_once('(')
                    .context("missing native target tensor type")?;
                dtype(ty.strip_suffix(')').context("invalid native target type")?)
                    .context("unknown native target type")?
            } else {
                tensor.ty
            };
            tensor.bytes(ty)?;
            ensure!(
                overrides
                    .get(name)
                    .is_none_or(|requested| dtype(requested) == Some(ty)),
                "native plan did not honor exact tensor override {name}"
            );
            ensure!(
                result.insert(name.into(), ty).is_none(),
                "duplicate native GGUF plan tensor"
            );
        }
        ensure!(
            result.len() == self.tensors.len(),
            "native GGUF dry run did not attest every tensor"
        );
        Ok(result)
    }

    pub(super) fn verify(&self, output: &Self, types: &Types, preset: u64) -> Result<()> {
        ensure!(
            output.ftype == Some(preset),
            "GGUF output preset metadata mismatch"
        );
        ensure!(
            output.architecture == self.architecture && output.tensors.len() == self.tensors.len(),
            "GGUF architecture or inventory mismatch"
        );
        for (name, t) in &self.tensors {
            let out = output
                .tensors
                .get(name)
                .with_context(|| format!("missing GGUF output tensor {name}"))?;
            ensure!(
                out.shape == t.shape,
                "GGUF tensor shape mismatch for {name}"
            );
            ensure!(
                out.ty == types[name],
                "GGUF tensor type mismatch for {name}"
            );
        }
        Ok(())
    }
}

pub(super) fn hf_inventory(value: &Value) -> Result<BTreeMap<String, Vec<u64>>> {
    let tensors: BTreeMap<String, Vec<u64>> =
        serde_json::from_value(value.clone()).context("invalid HF GGUF inventory")?;
    ensure!(
        (1..=32768).contains(&tensors.len())
            && tensors.iter().all(|(n, s)| valid_name(n)
                && (1..=4).contains(&s.len())
                && s.iter().all(|d| *d > 0)),
        "invalid HF GGUF tensor inventory"
    );
    Ok(tensors)
}
