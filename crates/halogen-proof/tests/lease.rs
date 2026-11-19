#![cfg(target_os = "linux")]
use std::{fs, process::Command};

#[test]
fn expired_lease_stops_an_uncooperative_runtime_without_touching_other_processes() {
    let dir = tempfile::tempdir().unwrap();
    let lease = dir.path().join("current");
    fs::write(&lease, "0").unwrap();
    let output = Command::new("/usr/bin/python3")
        .args([
            "-I",
            "-c",
            r#"
import os, runpy, signal, sys, time
m = runpy.run_path(sys.argv[1])
owner = os.fork()
if owner == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    me = os.getpid()
    if os.fork() == 0:
        m["watch"](me, sys.argv[2], 20, 0)
        os._exit(0)
    while True:
        time.sleep(0.1)
_, status = os.waitpid(owner, 0)
assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
"#,
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/lease.py"),
        ])
        .arg(lease)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}
