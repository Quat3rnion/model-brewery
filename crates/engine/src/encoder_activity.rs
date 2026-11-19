use crate::integrity::{file_digest, Stamp};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::OsStr,
    fs::OpenOptions,
    io::Read,
    os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    process::Command,
};

const SCHEMA: &str = "brewery-cupti-encoder-v1";
const CAPTURE_LIMIT: u64 = 1024;

pub(super) enum Capture {
    Disabled,
    Unavailable(String),
    Armed(Probe),
}

pub(super) struct Probe {
    path: PathBuf,
    collector: PathBuf,
    collector_sha256: String,
    collector_stamp: Stamp,
    exe_dev: u64,
    exe_ino: u64,
    uuid: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema: String,
    pid: u64,
    exe_dev: u64,
    exe_ino: u64,
    gpu_uuid: String,
    compute_major: u32,
    compute_minor: u32,
    counts: Counts,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Counts {
    ht3: u64,
    ht4: u64,
    i4r: u64,
    q4cp: u64,
}

impl Capture {
    pub(super) fn configure(
        cuda_convert: bool,
        cmd: &mut Command,
        exe: &Path,
        evidence: &Path,
    ) -> Self {
        if !cuda_convert {
            return Self::Disabled;
        }
        let Some(collector) = env::var_os("BREWERY_CUPTI_COLLECTOR") else {
            return Self::Disabled;
        };
        let result = env::var("BREWERY_CUDA_GPU_UUID")
            .ok()
            .context("BREWERY_CUDA_GPU_UUID must be a full GPU UUID")
            .and_then(|uuid| {
                Probe::arm(
                    cmd,
                    exe,
                    evidence,
                    Path::new(&collector),
                    &uuid,
                    env::var_os("CUDA_VISIBLE_DEVICES").as_deref(),
                    env::var_os("CUDA_INJECTION64_PATH").as_deref(),
                )
            });
        match result {
            Ok(probe) => Self::Armed(probe),
            Err(err) => Self::Unavailable(err.to_string()),
        }
    }

    pub(super) fn result(self, receipt: &Value, artifact: &Value) -> Option<Value> {
        Some(match self {
            Self::Disabled => return None,
            Self::Unavailable(reason) => json!({"status":"unavailable","reason":reason}),
            Self::Armed(probe) => match probe.finish(receipt, artifact) {
                Ok(result) => result,
                Err(err) => json!({"status":"not_observed","reason":err.to_string()}),
            },
        })
    }
}

impl Probe {
    fn arm(
        cmd: &mut Command,
        exe: &Path,
        evidence: &Path,
        collector: &Path,
        uuid: &str,
        visible: Option<&OsStr>,
        injection: Option<&OsStr>,
    ) -> Result<Self> {
        ensure!(
            valid_gpu_uuid(uuid),
            "BREWERY_CUDA_GPU_UUID must be a full GPU UUID"
        );
        ensure!(
            visible.is_none_or(|entry| entry == uuid),
            "existing CUDA_VISIBLE_DEVICES does not match the configured GPU UUID"
        );
        ensure!(
            injection.is_none_or(OsStr::is_empty),
            "existing CUDA_INJECTION64_PATH prevents workbench capture"
        );
        ensure!(
            collector.is_absolute()
                && collector.extension().is_some_and(|x| x == "so")
                && !collector.as_os_str().as_bytes().contains(&b':'),
            "BREWERY_CUPTI_COLLECTOR must name one absolute shared library"
        );
        let collector = collector
            .canonicalize()
            .context("cannot resolve CUPTI collector")?;
        ensure!(
            !collector.as_os_str().as_bytes().contains(&b':'),
            "CUPTI collector path cannot contain a colon"
        );
        let (identity, collector_stamp) = file_digest(&collector, "collector".into())?;
        let metadata = exe.metadata()?;
        let path = evidence.join("encoder-activity.json");
        ensure!(path.is_absolute(), "CUPTI evidence path must be absolute");
        ensure!(
            path.symlink_metadata()
                .is_err_and(|err| err.kind() == std::io::ErrorKind::NotFound),
            "CUPTI evidence path is not fresh"
        );
        cmd.env("CUDA_INJECTION64_PATH", &collector)
            .env("CUDA_VISIBLE_DEVICES", uuid)
            .env("BREWERY_CUPTI_CAPTURE", &path);
        Ok(Self {
            path,
            collector,
            collector_sha256: identity.sha256,
            collector_stamp,
            exe_dev: metadata.dev(),
            exe_ino: metadata.ino(),
            uuid: uuid.into(),
        })
    }

    fn finish(self, receipt: &Value, artifact: &Value) -> Result<Value> {
        self.collector_stamp.check(&self.collector)?;
        let fd = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&self.path)
            .context("collector did not leave a readable capture")?;
        let meta = fd.metadata()?;
        ensure!(
            meta.is_file() && meta.nlink() == 1 && meta.len() <= CAPTURE_LIMIT,
            "invalid encoder capture file"
        );
        let mut bytes = Vec::new();
        fd.take(CAPTURE_LIMIT + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 == meta.len(),
            "encoder capture changed during read"
        );
        let record: Record =
            serde_json::from_slice(&bytes).context("invalid encoder capture JSON")?;
        ensure!(
            receipt["exit_code"] == 0
                && record.schema == SCHEMA
                && record.pid != 0
                && Some(record.pid) == receipt["pid"].as_u64()
                && record.exe_dev == self.exe_dev
                && record.exe_ino == self.exe_ino,
            "capture does not identify the successful converter process"
        );
        ensure!(
            record.gpu_uuid == self.uuid && record.compute_major > 0 && record.compute_minor < 10,
            "capture does not identify the selected physical GPU"
        );
        ensure!(
            [
                record.counts.ht3,
                record.counts.ht4,
                record.counts.i4r,
                record.counts.q4cp
            ]
            .into_iter()
            .any(|count| count != 0),
            "no completed converter encoder kernel observed"
        );
        let output = artifact["bundle_sha256"]
            .as_str()
            .filter(|s| digest(s))
            .context("missing published conversion bundle digest")?;
        let executable = receipt["executable_sha256"]
            .as_str()
            .filter(|s| digest(s))
            .context("missing converter executable digest")?;
        Ok(json!({
            "status":"observed","schema":SCHEMA,"backend":"cuda",
            "pid":record.pid,"executable_sha256":executable,"output_bundle_sha256":output,
            "gpu_uuid":record.gpu_uuid,"compute_capability":[record.compute_major,record.compute_minor],
            "completed_encoder_activity":record.counts,"capture":self.path,
            "capture_sha256":format!("{:x}", Sha256::digest(bytes)),
            "collector":self.collector,"collector_sha256":self.collector_sha256,
            "scope":"completed encoder kernels observed; not proof of full GPU coverage or release"
        }))
    }
}

fn lowercase_hex(byte: u8) -> bool {
    matches!(byte, b'0'..=b'9' | b'a'..=b'f')
}

fn valid_gpu_uuid(uuid: &str) -> bool {
    uuid.strip_prefix("GPU-").is_some_and(|value| {
        value.len() == 36
            && value.bytes().enumerate().all(|(i, b)| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    b == b'-'
                } else {
                    lowercase_hex(b)
                }
            })
    })
}

fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(lowercase_hex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrity::identify;
    use std::fs;

    #[test]
    fn capture_rejects_foreign_pid_and_zero_encodes() {
        // A test-only receipt, not hardware evidence. Probe::arm does not
        // execute or load either file in this test.
        let dir = tempfile::tempdir().unwrap();
        let evidence = dir.path().canonicalize().unwrap();
        let exe = env::current_exe().unwrap().canonicalize().unwrap();
        let collector = evidence.join("collector.so");
        fs::write(&collector, b"test collector").unwrap();
        let uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let mut cmd = Command::new(&exe);
        let probe = Probe::arm(&mut cmd, &exe, &evidence, &collector, uuid, None, None).unwrap();
        let hash = "a".repeat(64);
        let artifact = json!({"bundle_sha256":hash});
        let receipt = json!({"exit_code":0,"pid":10,"executable_sha256":hash});
        let mut record = json!({"schema":SCHEMA,"pid":11,"exe_dev":probe.exe_dev,
            "exe_ino":probe.exe_ino,"gpu_uuid":uuid,"compute_major":12,"compute_minor":0,
            "counts":{"ht3":1,"ht4":0,"i4r":0,"q4cp":0}});
        fs::write(&probe.path, record.to_string()).unwrap();
        let output = Capture::Armed(probe).result(&receipt, &artifact).unwrap();
        assert_eq!(output["status"], "not_observed");
        assert!(output["reason"]
            .as_str()
            .unwrap()
            .contains("converter process"));
        let path = evidence.join("encoder-activity.json");
        fs::remove_file(&path).unwrap();
        let probe = Probe::arm(&mut cmd, &exe, &evidence, &collector, uuid, None, None).unwrap();
        record["pid"] = json!(10);
        record["counts"]["ht3"] = json!(0);
        fs::write(path, record.to_string()).unwrap();
        let output = Capture::Armed(probe).result(&receipt, &artifact).unwrap();
        assert_eq!(output["status"], "not_observed");
        assert!(output["reason"].as_str().unwrap().contains("no completed"));
    }

    #[test]
    #[ignore = "requires installed CUDA structural fixture, collector, UUID and GPU/build gates"]
    fn installed_cuda_fixture_preserves_json_and_rejects_decode_only() {
        let fixture = PathBuf::from(env::var_os("BREWERY_TEST_CUDA_FIXTURE").unwrap())
            .canonicalize()
            .unwrap();
        let out = PathBuf::from(env::var_os("BREWERY_TEST_ACTIVITY_DIR").unwrap())
            .canonicalize()
            .unwrap();
        let root = out.join("fixture");
        for (mode, decode_only) in [("encode", false), ("decode", true)] {
            let evidence = out.join(mode);
            fs::create_dir(&evidence).unwrap();
            let mut cmd = Command::new(&fixture);
            cmd.arg("--root").arg(&root).args([
                "--backend",
                "cuda",
                "--ht-backend",
                "cuda",
                "--q4cp-backend",
                "cuda",
                "--decode-backend",
                "cuda",
            ]);
            if decode_only {
                cmd.arg("--verify-only");
            }
            let capture = Capture::configure(true, &mut cmd, &fixture, &evidence);
            assert!(matches!(capture, Capture::Armed(_)));
            let (report, receipt) =
                crate::native_json(&mut cmd, &evidence.join("stderr.log")).unwrap();
            assert_eq!(report["verified_identical_hgn_bytes"], true);
            let gpu = root.join("cuda");
            let identity = identify(&gpu, false).unwrap();
            let activity = capture
                .result(&receipt, &identity.artifact(&gpu, "hgn"))
                .unwrap();
            crate::write_json(&evidence.join("activity-result.json"), &activity).unwrap();
            if decode_only {
                assert_eq!(activity["status"], "not_observed", "{activity}");
                assert!(
                    activity["reason"]
                        .as_str()
                        .unwrap()
                        .contains("no completed"),
                    "{activity}"
                );
            } else {
                assert_eq!(activity["status"], "observed", "{activity}");
                assert_eq!(activity["compute_capability"], json!([12, 0]));
                assert_eq!(activity["pid"], receipt["pid"]);
                assert_eq!(activity["executable_sha256"], receipt["executable_sha256"]);
                assert_eq!(activity["output_bundle_sha256"], identity.bundle_sha256);
                for codec in ["ht3", "ht4", "q4cp"] {
                    assert!(
                        activity["completed_encoder_activity"][codec]
                            .as_u64()
                            .unwrap()
                            > 0
                    );
                }
            }
        }
    }
}
