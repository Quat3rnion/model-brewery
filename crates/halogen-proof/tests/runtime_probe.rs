#![cfg(target_os = "linux")]

use std::{fs, net::TcpListener, os::unix::fs::MetadataExt, process::Command};

#[test]
fn a_ready_lookalike_does_not_prove_a_bundle_with_an_unmapped_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let engine = dir.path().join("flash_serve");
    fs::copy("/usr/bin/python3", &engine).unwrap();
    let model = dir.path().join("model.hgn");
    let lookup = dir.path().join("lookup.hgn");
    for path in [&model, &lookup] {
        fs::write(path, vec![0u8; 8192]).unwrap();
    }
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let fixture = r#"
import mmap, socket, sys
with open(sys.argv[2], 'rb') as f:
    mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
s = socket.socket()
s.bind(('127.0.0.1', int(sys.argv[4])))
s.listen()
while True:
    c, _ = s.accept()
    with c:
        if c.recv(5) == b'PING\n':
            c.sendall(b'PONG\n')
"#;
    let mut child = Command::new(&engine)
        .args(["-c", fixture, "--ck"])
        .arg(&model)
        .args(["--port", &port])
        .spawn()
        .unwrap();
    let entries = |files: &[&std::path::Path]| {
        let entries: Vec<_> = files
            .iter()
            .map(|p| {
                let m = fs::metadata(p).unwrap();
                serde_json::json!({
                    "path": p,
                    "major": libc::major(m.dev()),
                    "minor": libc::minor(m.dev()),
                    "inode": m.ino(),
                    "bytes": m.len(),
                })
            })
            .collect();
        serde_json::to_string(&entries).unwrap()
    };
    let probe = |files: &[&std::path::Path]| {
        Command::new("/usr/bin/python3")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/observe.py"))
            .arg(&engine)
            .arg(std::process::id().to_string())
            .arg(&port)
            .arg("2")
            .arg(entries(files))
            .output()
            .unwrap()
    };
    let positive_control = probe(&[&model]);
    let missing_sidecar = probe(&[&model, &lookup]);
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(
        positive_control.status.success(),
        "{}",
        String::from_utf8_lossy(&positive_control.stderr)
    );
    assert!(!missing_sidecar.status.success());
    assert!(missing_sidecar.stdout.is_empty());
    assert!(String::from_utf8_lossy(&missing_sidecar.stderr).contains("not fully mapped read-only"));
}
