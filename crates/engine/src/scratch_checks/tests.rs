use super::{run_with_libraries, Job, Progress};
use brewery_worker::scratch_eval::RUNNER_LIB_DIR as LIB_DIR;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env, fs, io,
    path::Path,
    process::{Command, Stdio},
};

const LIB: &str = "libllama.so.0";
const FIXTURE_SHA: &str = "BREWERY_TEST_CHECKS_LIBRARY_SHA256";

fn cc(args: &[&str], source: &str) {
    use std::io::Write;
    let mut child = Command::new("cc")
        .args(args)
        .env_remove("LD_LIBRARY_PATH")
        .args(["-x", "c", "-"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn scratch_checks_run_offline_and_bind_the_native_report() {
    // Re-execute this test to give the job an isolated worker environment.
    if let Ok(lib_sha) = env::var(FIXTURE_SHA) {
        let job: Job = serde_json::from_reader(io::stdin()).unwrap();
        let cwd = env::current_dir().unwrap();
        let progress = Progress::new(&cwd, job.inputs.iter().map(|i| &i.path));
        let report = run_with_libraries(&job, &cwd, &progress, &[(LIB, &lib_sha)])
            .unwrap_or_else(|err| panic!("{err:#}"));
        assert_eq!(
            report["revision_sha256"].as_str(),
            job.inputs[3].sha256.as_deref()
        );
        assert_eq!(report["metrics"][0]["child"], 2.0);
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let configured = root.join("configured");
    let runpath = root.join("runpath");
    for (path, value) in [(&configured, "configured"), (&runpath, "wrong")] {
        fs::create_dir(path).unwrap();
        cc(
            &[
                "-shared",
                "-fPIC",
                "-Wl,-soname,libllama.so.0",
                "-o",
                path.join(LIB).to_str().unwrap(),
            ],
            &format!("const char *fixture(void) {{ return \"{value}\"; }}"),
        );
    }
    let runner = root.join("runner");
    cc(
        &[
            "-Wl,--enable-new-dtags",
            &format!("-Wl,-rpath,{}", runpath.display()),
            "-Wl,--no-as-needed",
            runpath.join(LIB).to_str().unwrap(),
            "-o",
            runner.to_str().unwrap(),
        ],
        r#"
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
extern const char *fixture(void);
int main(void) {
    Dl_info info;
    if (!dladdr(fixture, &info)) return 1;
    printf("%s\n%s\n", fixture(), info.dli_fname);
}
"#,
    );
    let baseline = Command::new(&runner).env_clear().output().unwrap();
    assert!(baseline.status.success());
    assert_eq!(
        String::from_utf8(baseline.stdout).unwrap(),
        format!("wrong\n{}\n", runpath.join(LIB).display()),
    );
    let script = root.join("check.py");
    fs::write(
        &script,
        r#"
import argparse, hashlib, json, os, pathlib, subprocess
p = argparse.ArgumentParser()
for arg in ("runner", "cache", "parent-gguf", "child-gguf", "parent-safetensors", "child-safetensors"):
    p.add_argument("--" + arg, required=True)
p.add_argument("--threads", type=int, required=True)
a = p.parse_args()
assert a.threads == 2
assert pathlib.Path(os.environ["BREWERY_NATIVE_PROGRESS"]).name == "progress.json"
value, loaded = subprocess.check_output([a.runner], text=True).splitlines()
assert value == "configured", value
assert pathlib.Path(loaded).parent.parent == pathlib.Path(a.runner).parent, loaded
pins = json.loads(pathlib.Path(a.cache, "manifest.json").read_text())
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
digest = sha(a.child_safetensors) if not pathlib.Path(a.cache, "bad").exists() else "0" * 64
print(json.dumps(dict(schema="brewery.scratch-behavior.v1", qualification="not release-qualified",
    runner_sha256=sha(a.runner), revision_sha256=digest, parent_sha256=sha(a.parent_safetensors),
    parent_gguf_sha256=sha(a.parent_gguf), revision_gguf_sha256=sha(a.child_gguf),
    fixture_sha256=pins, parent_generation=" parent", child_generation=" child",
    metrics=[dict(name="perplexity", parent=1., child=2., better="lower")])))
"#,
    )
    .unwrap();
    fs::write(
        root.join("manifest.json"),
        include_str!("../../../eval/data/edit-checks/manifest.json"),
    )
    .unwrap();
    let digest = |path: &Path| format!("{:x}", Sha256::digest(fs::read(path).unwrap()));
    let inputs: Vec<_> = ["gguf", "gguf", "safetensors", "safetensors"]
        .into_iter()
        .enumerate()
        .map(|(i, kind)| {
            let path = root.join(format!("model-{i}.{kind}"));
            fs::write(&path, format!("GGUF{i}")).unwrap();
            json!({"host":"local","kind":kind,"path":path,"sha256":digest(&path)})
        })
        .collect();
    let lib_sha = digest(&configured.join(LIB));
    let run = |name, runner_sha, lib_dir: Option<&Path>| {
        let work = root.join(name);
        fs::create_dir(&work).unwrap();
        let mut cmd = Command::new(env::current_exe().unwrap());
        cmd.env_clear();
        if let Some(dir) = lib_dir {
            cmd.env(LIB_DIR, dir);
        }
        let mut child = cmd
            .args([
                "--exact",
                "scratch_checks::tests::scratch_checks_run_offline_and_bind_the_native_report",
                "--nocapture",
            ])
            .env(FIXTURE_SHA, &lib_sha)
            .env("LD_LIBRARY_PATH", &runpath)
            .env("BREWERY_CPU_THREADS", "2")
            .env("BREWERY_SCRATCH_CHECKS_SCRIPT", &script)
            .env("BREWERY_SCRATCH_CHECKS_RUNNER", &runner)
            .env("BREWERY_SCRATCH_CHECKS_RUNNER_SHA256", runner_sha)
            .env("BREWERY_SCRATCH_CHECKS_CACHE", &root)
            .current_dir(work)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        serde_json::to_writer(
            child.stdin.take().unwrap(),
            &json!({"kind":"scratch_checks","inputs":inputs,"options":{}}),
        )
        .unwrap();
        child.wait_with_output().unwrap()
    };
    let output = run("wrong-runner", "0".repeat(64), Some(&configured));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("scratch checks runner SHA-256 mismatch"),
        "{stderr}"
    );
    let output = run("verified", digest(&runner), Some(&configured));
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        fs::read_to_string(root.join("verified/scratch-checks.log")).unwrap_or_default(),
    );
    fs::write(root.join("bad"), b"").unwrap();
    let output = run("wrong-report", digest(&runner), Some(&configured));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("scratch checks report does not match"),
        "{stderr}"
    );
    let output = run("no-setting", digest(&runner), None);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("{LIB_DIR} is required")),
        "{stderr}"
    );
    fs::remove_file(configured.join(LIB)).unwrap();
    let output = run("missing-library", digest(&runner), Some(&configured));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("open scratch checks library libllama.so.0"),
        "{stderr}"
    );
    fs::copy(runpath.join(LIB), configured.join(LIB)).unwrap();
    let output = run("wrong-library", digest(&runner), Some(&configured));
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("scratch checks library libllama.so.0 SHA-256 mismatch"),
        "{stderr}"
    );
}
