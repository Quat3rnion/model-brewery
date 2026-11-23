//! Bounded, uncalibrated EXL3 layer encoding. The worker never exposes the
//! source to the helper: it receives only a private, transposed FP32 block.
use crate::{Job, integrity, progress::Progress};
use anyhow::{Context, Result, bail, ensure};
use brewery_worker::{
    exl3_layer::{self as wire, Codebook, DIM, Spec},
    weight_source,
};
use serde_json::{Value, json};
use std::{
    env, fs,
    io::{self, Read, Seek},
    os::unix::{fs::PermissionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Safetensors<'a> {
    payload: &'a [u8],
    header: weight_source::Header,
}

impl<'a> Safetensors<'a> {
    fn parse(bytes: &'a [u8], limit: u64) -> Result<Self> {
        let header = weight_source::Header::read(bytes, bytes.len() as u64, limit)?;
        let payload = &bytes[header.payload_start as usize..];
        Ok(Self { payload, header })
    }

    fn tensor(&self, name: &str, shape: &[u64], dtypes: &[&str]) -> Result<(&str, &'a [u8])> {
        let t = self
            .header
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name:?}"))?;
        ensure!(
            t.shape == shape && dtypes.contains(&t.dtype.as_str()),
            "unexpected tensor shape or dtype for {name:?}"
        );
        let [start, end] = t.data_offsets;
        Ok((&t.dtype, &self.payload[start as usize..end as usize]))
    }
}

pub(crate) fn half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let magnitude = match exponent {
        0 => f64::from(fraction) * 2f64.powi(-24),
        31 if fraction == 0 => f64::INFINITY,
        31 => f64::NAN,
        _ => f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25),
    };
    if bits >> 15 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn floats(dtype: &str, bytes: &[u8]) -> Result<Vec<f64>> {
    let values: Vec<_> = match dtype {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f64::from(f32::from_le_bytes(*v)))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| half(u16::from_le_bytes(*v)))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*v)) << 16)))
            .collect(),
        _ => bail!("unexpected dense dtype"),
    };
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite tensor value"
    );
    Ok(values)
}

fn f32_bytes(values: impl IntoIterator<Item = f64>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| (v as f32).to_le_bytes())
        .collect()
}

fn run_helper(helper: &Path, stamp: &integrity::Stamp, cwd: &Path, args: &[&str]) -> Result<()> {
    stamp.check(helper)?;
    let stderr = tempfile::tempfile_in(cwd)?;
    let mut cmd = Command::new(helper);
    cmd.current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr.try_clone()?);
    // The worker owns the process tree; bound each subprocess's wall time
    // and files without setting a GPU-incompatible address-space cap.
    unsafe {
        cmd.pre_exec(|| {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::getrlimit(libc::RLIMIT_FSIZE, &mut limit) != 0 {
                return Err(io::Error::last_os_error());
            }
            limit.rlim_cur = limit.rlim_cur.min(64 << 10);
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd
        .spawn()
        .context("cannot start native EXL3 checkpoint helper")?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                let mut log = Vec::new();
                let mut stderr = stderr;
                stderr.rewind()?;
                stderr.take(1024).read_to_end(&mut log)?;
                bail!(
                    "native EXL3 {} failed ({status}): {}",
                    args[0],
                    String::from_utf8_lossy(&log).trim()
                );
            }
            break;
        }
        if start.elapsed() >= Duration::from_secs(30) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("native EXL3 {} exceeded its time bound", args[0]);
        }
        thread::sleep(Duration::from_millis(20));
    }
    stamp.check(helper)?;
    Ok(())
}

fn hadamard(values: &mut [f64; DIM]) {
    for s in [1, 2, 4, 8, 16, 32, 64] {
        for start in (0..DIM).step_by(2 * s) {
            for i in start..start + s {
                let a = values[i];
                let b = values[i + s];
                values[i] = a + b;
                values[i + s] = a - b;
            }
        }
    }
}

fn codebook(state: u32, book: Codebook) -> f64 {
    let value = if book == Codebook::Mul1 {
        let h = state.wrapping_mul(0x83dcd12d);
        let n = 1024 + h.to_le_bytes().into_iter().map(u32::from).sum::<u32>();
        f64::from(n) * half(0x1eee) + half(0xc931)
    } else {
        let h = if book == Codebook::Mcg {
            state.wrapping_mul(0xcbac1fed)
        } else {
            state.wrapping_mul(89226354).wrapping_add(64248484)
        };
        let h = (h & 0x8fff8fff) ^ 0x3b603b60;
        half(h as u16) + half((h >> 16) as u16)
    };
    if value == 0. {
        return value;
    }
    // Reference half-add/half-FMA rounding, independent of the native binary.
    let scale = 2f64.powi((value.abs().log2().floor() as i32 - 10).max(-24));
    (value / scale).round_ties_even() * scale
}

fn reconstruct(path: &Path, spec: &Spec) -> Result<Vec<f64>> {
    let mut bytes = Vec::new();
    weight_source::open_regular(path)?
        .take(wire::MAX_OUTPUT + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= wire::MAX_OUTPUT,
        "invalid EXL3 output extent"
    );
    let sf = Safetensors::parse(&bytes, 4096)?;
    ensure!(
        sf.header.entries.len() == 3 + usize::from(spec.codebook.marker().is_some()),
        "EXL3 artifact must contain only one bias-free layer group"
    );
    if let Some(marker) = spec.codebook.marker() {
        let name = format!("layer.{}", spec.codebook.name());
        let (_, data) = sf.tensor(&name, &[], &["I32"])?;
        ensure!(
            data == marker.to_le_bytes(),
            "incorrect EXL3 codebook marker"
        );
    }
    let (_, su) = sf.tensor("layer.suh", &[DIM as u64], &["F16"])?;
    let (_, sv) = sf.tensor("layer.svh", &[DIM as u64], &["F16"])?;
    let su = floats("F16", su)?;
    let sv = floats("F16", sv)?;
    let rate = usize::from(spec.twice_bits);
    let (_, packed) = sf.tensor("layer.trellis", &[8, 8, (8 * rate) as u64], &["I16"])?;
    let mut w = vec![[0.; DIM]; DIM];
    for (tile, bytes) in packed.chunks_exact(16 * rate).enumerate() {
        let ring: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| u32::from_le_bytes(*b).to_be_bytes())
            .collect();
        let width = ring.len() * 8;
        for pos in 0..256 {
            let start = (width + (pos + 1) * rate / 2 - 16) % width;
            let mut state = 0;
            // The ring is MSB-first after converting the little-endian words.
            for bit in start..start + 16 {
                let bit = bit % width;
                state = state * 2 + u32::from((ring[bit / 8] >> (7 - bit % 8)) & 1);
            }
            let (lane, el) = (pos / 8, pos % 8);
            let r = tile / 8 * 16 + (lane % 4) * 2 + (el & 1) + (el & 2) * 4;
            let c = tile % 8 * 16 + lane / 4 + (el & 4) * 2;
            w[r][c] = codebook(state, spec.codebook);
        }
    }
    for row in &mut w {
        hadamard(row);
    }
    for c in 0..DIM {
        let mut column = std::array::from_fn(|r| w[r][c]);
        hadamard(&mut column);
        for r in 0..DIM {
            w[r][c] = column[r] * su[r] * sv[c] / DIM as f64;
        }
    }
    Ok(w.into_iter().flatten().collect())
}

pub(crate) fn run(job: &Job, work: &Path, progress: &Progress) -> Result<Value> {
    let spec = Spec::parse(&job.options)?;
    ensure!(
        job.prompt.is_none() && job.inputs.len() == 1,
        "invalid EXL3 layer job"
    );
    let input = &job.inputs[0];
    ensure!(
        input.kind == "safetensors"
            && input.source_host.as_ref() == Some(&input.host)
            && input.integrity_basis.is_none()
            && env::var_os("BREWERY_EXPECTED_SOURCE_SHA256").is_none(),
        "EXL3 requires one direct standalone SafeTensors source on an ordinary worker"
    );
    ensure!(
        env::var(weight_source::GATE).as_deref() == Ok("1"),
        "EXL3 SafeTensors is not enabled"
    );
    let limit = weight_source::verify_limit(env::var(weight_source::VERIFY_BYTES).ok().as_deref())?
        .min(wire::MAX_SOURCE);
    weight_source::bounded_inventory(&input.path, &input.kind, limit)?;
    let source = integrity::Identity::weight_manifest(input)?;
    ensure!(
        source.sha256 == spec.source_sha256 && source.bytes <= limit,
        "EXL3 source fingerprint or extent changed"
    );
    let (dtype, dense) = {
        let snapshot = source.snapshot_single_weight_member(limit, progress)?;
        let sf = Safetensors::parse(&snapshot, 4 << 20)?;
        let (dtype, bytes) = sf.tensor(&spec.tensor, &[DIM as u64; 2], &["F16", "BF16", "F32"])?;
        (dtype.to_owned(), floats(dtype, bytes)?)
    };
    source.check()?;
    let original: Vec<_> = (0..DIM * DIM)
        .map(|i| dense[(i % DIM) * DIM + i / DIM])
        .collect();

    let tool = PathBuf::from(env::var_os(wire::HELPER).context("EXL3 helper is not configured")?);
    let meta = integrity::canonical(&tool)?;
    ensure!(
        meta.is_file() && meta.permissions().mode() & 0o111 != 0 && meta.len() <= 128 << 20,
        "expected bounded canonical EXL3 executable"
    );
    let (binary, stamp) = integrity::file_digest(&tool, "exl3-checkpoint".into())?;
    let output = work.join(wire::FILE);
    match output.symlink_metadata() {
        Ok(_) => bail!("EXL3 output already exists"),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let scratch = tempfile::Builder::new()
        .prefix(wire::SCRATCH)
        .tempdir_in(work)?;
    let root = scratch.path();
    fs::write(
        root.join("weights.f32"),
        f32_bytes(original.iter().copied()),
    )?;
    {
        let _phase = progress.start("convert", "items", None);
        run_helper(
            &tool,
            &stamp,
            root,
            &[
                "encode",
                spec.codebook.name(),
                &spec.twice_bits.to_string(),
                "layer",
                "weights.f32",
                wire::FILE,
            ],
        )?;
    }
    let _phase = progress.start("verify_output", "items", None);
    let scratch_output = root.join(wire::FILE);
    ensure!(
        fs::symlink_metadata(&scratch_output)
            .is_ok_and(|m| m.is_file() && m.len() <= wire::MAX_OUTPUT),
        "invalid EXL3 output extent"
    );
    let (artifact, packed_stamp) = integrity::file_digest(&scratch_output, wire::FILE.into())?;
    let reconstructed = reconstruct(&scratch_output, &spec)?;
    let (mut energy, mut squared_error, mut max_abs) = (0., 0., 0_f64);
    for (&actual, &expected) in reconstructed.iter().zip(&original) {
        let delta = (actual - expected).abs();
        energy += expected * expected;
        squared_error += delta * delta;
        max_abs = max_abs.max(delta);
    }
    ensure!(
        energy != 0. || squared_error == 0.,
        "zero weights reconstructed as nonzero"
    );
    let relative = if energy == 0. {
        0.
    } else {
        (squared_error / energy).sqrt()
    };
    ensure!(
        relative.is_finite() && relative <= 1.,
        "EXL3 relative L2 error exceeds 1"
    );
    let mut backend_max = 0_f64;
    // Probe three fixed, distinct vectors on the requested device against
    // dense products of the independently reconstructed full layer.
    for probe in 0..3 {
        let x: [f64; DIM] = std::array::from_fn(|i| match probe {
            0 => f64::from(i == 17),
            1 => f64::from(i == 73),
            _ => f64::from(((i * 17 + 11) % 31) as i32 - 15) / 16.,
        });
        let xname = format!("x{probe}.f32");
        let yname = format!("y{probe}.f32");
        fs::write(root.join(&xname), f32_bytes(x))?;
        run_helper(
            &tool,
            &stamp,
            root,
            &[
                "forward",
                spec.backend.name(),
                wire::FILE,
                "layer",
                &xname,
                &yname,
            ],
        )?;
        let mut yfile = weight_source::open_regular(&root.join(&yname))?;
        let mut data = [0; DIM * 4];
        ensure!(
            yfile.metadata()?.len() == data.len() as u64,
            "invalid EXL3 forward output extent"
        );
        yfile.read_exact(&mut data)?;
        let y = floats("F32", &data)?;
        let scale = reconstructed.iter().map(|v| v.abs()).fold(0_f64, f64::max);
        for (c, observed) in y.into_iter().enumerate() {
            let expected = (0..DIM)
                .map(|r| x[r] * reconstructed[r * DIM + c])
                .sum::<f64>();
            let delta = (observed - expected).abs();
            ensure!(
                delta <= (scale * 2e-4).max(1e-44),
                "requested EXL3 backend disagrees with independent dense reconstruction"
            );
            backend_max = backend_max.max(delta);
        }
    }
    packed_stamp.check(&scratch_output)?;
    source.check()?;
    let mut report = serde_json::to_value(&spec)?;
    report["schema"] = json!(wire::SCHEMA);
    report["verification"] = json!("independent-decode-and-forward");
    report["module"] = json!("layer");
    report["source_dtype"] = json!(dtype);
    report["relative_l2_error"] = json!(relative);
    report["max_abs_error"] = json!(max_abs);
    report["backend_max_abs_error"] = json!(backend_max);
    let result = json!({
        "kind":wire::JOB, "host":input.host, "source_format":"safetensors",
        "source_sha256":source.sha256, "inputs":[source.input(input)],
        "verified":true, "release_qualified":false, "report":report,
        "backend_receipt":{"tool":wire::HELPER,"executable_sha256":binary.sha256},
        "artifact":{"kind":wire::KIND,"path":output,"sha256":artifact.sha256,"bytes":artifact.bytes}
    });
    spec.check_receipt(&input.host, &result)?;
    fs::File::open(&scratch_output)?.sync_all()?;
    fs::hard_link(&scratch_output, &output).context("publish new EXL3 layer artifact")?;
    if let Err(error) = fs::File::open(work).and_then(|dir| dir.sync_all()) {
        let _ = fs::remove_file(&output);
        return Err(error).context("sync EXL3 artifact directory");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn fixture(value: f32) -> Vec<u8> {
        let len = DIM * DIM * 4;
        let header = json!({
            "proj.weight": {"dtype": "F32", "shape": [DIM, DIM], "data_offsets": [0, len]}
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(std::iter::repeat_n(value.to_le_bytes(), DIM * DIM).flatten());
        bytes
    }

    #[test]
    fn exl3_encoder_input_uses_verified_bytes_after_source_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().canonicalize().unwrap().join("model.safetensors");
        let original = fixture(1.);
        fs::write(&path, &original).unwrap();
        let bytes = original.len();
        let sha256 = format!("{:x}", Sha256::digest(&original));
        let bundle = format!("model.safetensors\0{bytes}\0{sha256}\n");
        let input = serde_json::from_value(json!({
            "path": path, "kind": "safetensors", "host": "local",
            "sha256": sha256, "bytes": bytes,
            "bundle_sha256": format!("{:x}", Sha256::digest(bundle)),
            "files": [{"path": "model.safetensors", "sha256": sha256, "bytes": bytes}]
        }))
        .unwrap();
        let source = integrity::Identity::weight_manifest(&input).unwrap();
        let progress = Progress::new(dir.path(), [&path]);
        assert!(
            source
                .snapshot_single_weight_member(bytes as u64 - 1, &progress)
                .is_err()
        );
        let snapshot = source
            .snapshot_single_weight_member(bytes as u64, &progress)
            .unwrap();

        // No scheduling dependency: the write happens after the digest check.
        fs::write(&path, fixture(2.)).unwrap();
        assert_eq!(snapshot, original);
        let sf = Safetensors::parse(&snapshot, 4 << 20).unwrap();
        let (dtype, data) = sf
            .tensor("proj.weight", &[DIM as u64; 2], &["F32"])
            .unwrap();
        assert_eq!(floats(dtype, data).unwrap(), vec![1.; DIM * DIM]);
    }
}
