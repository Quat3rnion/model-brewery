"""Verify a pinned, reproducible install and the real helper sandbox."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from tempfile import mkdtemp

from managed_weight_smoke import TENSOR, checksum, matrix, save

REPO = Path(__file__).resolve().parents[1]
PACKAGER = ["/usr/bin/python3", "-IB", REPO / "tools/brewery-weight-runtime.py"]
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def command(args, *, cwd=None, env=ENV, ok=True, pass_fds=(), error=None):
    p = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True,
                       timeout=25, pass_fds=pass_fds)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stderr)
    assert error is None or error in p.stderr, (args, p.stderr)
    return p.stdout.strip()


def pinned_source_smoke(edit, root):
    work = root / ".weight-safetensors-pin-smoke"
    work.mkdir()
    host = root / "host-source"
    host.mkdir()
    source = host / "original.safetensors"
    head = b'{"layer.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    head += b" " * (-len(head) % 8)
    original = struct.pack("<Q", len(head)) + head + struct.pack("<f", 7)
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    manifest = work / "source-manifest.json"
    save(manifest, [{"path": source.name, "bytes": len(original),
                     "sha256": digest, "chunk_sha256": [digest]}])
    fixture = "/weight-source/source.safetensors"
    pin = ["--source-sha256", digest]
    proof = ["--source-manifest", manifest]
    args = ["inspect", fixture, *pin, "--tensor=layer.weight", "--index=[0]", *proof]
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        saved = host / "saved.safetensors"
        source.rename(saved)
        source.write_bytes(original[:-4] + struct.pack("<f", 6))

        def run(rest, leaf=source.name, member_fd=fd, **kwargs):
            return command([edit, "--pinned-safetensors", "1", str(member_fd), leaf, "--", *rest],
                           cwd=work, pass_fds=(member_fd,), **kwargs)

        assert json.loads(run(args))["value"] == 7
        run(args, leaf="../original.safetensors", ok=False)
        run(args, leaf="not-in-manifest.safetensors", ok=False)
        writable = os.open(saved, os.O_RDWR)
        try:
            run(args, member_fd=writable, ok=False)
        finally:
            os.close(writable)

        save(work / "request.json", [{"tensor": "layer.weight", "index": [0],
                                      "op": "set", "value": 8}])
        proposal = json.loads(run(["propose", fixture, "request.json", *pin, *proof]))
        save(work / "proposal.json", proposal)
        apply = ["apply", fixture, "proposal.json"]
        run([*apply, "/weight-source/shadow.safetensors", *proof], ok=False)
        out = work / "edited.safetensors"
        run([*apply, out, *proof])
        expected = original[:-4] + struct.pack("<f", 8)
        assert out.read_bytes() == expected
        run([*apply, out, *proof], ok=False)
        assert out.read_bytes() == expected
        assert saved.read_bytes() == original
    finally:
        os.close(fd)


def confined_source_smoke(release, changed_release, root):
    name = "brewery-weight-edit-confined.py"
    edit = release / name
    state = root / "sandbox-state"
    state.mkdir()
    work = state / ".weight-safetensors-smoke"
    work.mkdir()
    original = root / "fixture.safetensors"
    header = b'{"layer.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    header += b" " * (-len(header) % 8)
    sample = struct.pack("<Q", len(header)) + header + struct.pack("<f", 7)
    original.write_bytes(sample)
    source = work / "source.safetensors"
    source.write_bytes(sample)
    digest = hashlib.sha256(sample).hexdigest()
    manifest = work / "source-manifest.json"
    save(manifest, [{"path": source.name, "bytes": len(sample), "sha256": digest}])
    # Match the Hy4 service confinement, including an exact read-only model
    # member and no /usr/bin/sh or nested bubblewrap.
    sandbox = [
        "/usr/bin/bwrap", "--unshare-user", "--unshare-all", "--disable-userns", "--cap-drop", "ALL",
        "--die-with-parent", "--new-session", "--clearenv", "--ro-bind", "/usr/lib", "/usr/lib",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
        "--ro-bind", Path("/usr/bin/python3").resolve(), "/usr/bin/python3",
        "--proc", "/proc", "--remount-ro", "/proc",
        "--dev-bind", "/dev/null", "/dev/null", "--dev-bind", "/dev/urandom", "/dev/urandom",
        "--ro-bind", release, release, "--ro-bind", changed_release, changed_release,
        "--ro-bind", original, original, "--bind", state, state, "--chdir", work,
        "--remount-ro", "/", "--",
    ]

    def run(args, **kwargs):
        return command([*sandbox, edit, *args], **kwargs)

    status = json.loads(run(["--capabilities"]))
    assert status == {"schema": "brewery.weight-runtime.v1",
                      "jobs": ["weight_inspect", "weight_propose", "weight_apply"],
                      "formats": ["safetensors", "hf"], "visual": True,
                      "packed": False, "proof_source_fds": False}
    command([edit, "--capabilities"], ok=False, error="requires the restricted worker sandbox")
    command([*sandbox, changed_release / name, "--capabilities"], ok=False,
            error="weight runtime file does not match manifest: weight_edit.py")
    pin = ["--source-sha256", digest]
    proof = ["--source-manifest", manifest]
    inspect = [*pin, "--tensor=layer.weight", "--index=[0]", *proof]
    assert json.loads(run(["inspect", source.name, *inspect]))["value"] == 7
    run(["inspect", original, *inspect], ok=False)
    source.unlink()
    source.symlink_to(original)
    run(["inspect", source.name, *inspect], ok=False)
    source.unlink()
    source.write_bytes(sample)
    save(work / "request.json", [{"tensor": "layer.weight", "index": [0],
                                  "op": "set", "value": 8}])
    proposal = json.loads(run(["propose", source.name, "request.json", *pin, *proof]))
    save(work / "proposal.json", proposal)
    output = work / "output.safetensors"
    run(["apply", source.name, "proposal.json", original, *proof], ok=False)
    run(["apply", source.name, "proposal.json", output, *proof])
    assert output.read_bytes() == sample[:-4] + struct.pack("<f", 8)
    run(["apply", source.name, "proposal.json", output, *proof], ok=False)
    assert source.read_bytes() == original.read_bytes() == sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-python", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(mkdtemp(prefix="weight-runtime-smoke.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    dependencies = ["--gguf-python", args.gguf_python.resolve(strict=True)]
    pin = command([*PACKAGER, "fingerprint", *dependencies])

    def install(path, fingerprint, ok=True):
        return command([*PACKAGER, "install", *dependencies, "--bin-dir", path,
                        "--dependency-sha256", fingerprint], ok=ok)

    a, b = root / "release-a", root / "release-b"
    for path in (a, b):
        path.mkdir()
    install(a, "0" * 64, ok=False)
    assert not list(a.iterdir()), "unapproved dependencies published files"
    manifest = install(a, pin)
    assert install(b, pin) == manifest
    for name in ("brewery-weight-edit", "brewery-weight-tune", "brewery-weight-edit-confined.py"):
        assert (a / name).read_bytes() == (b / name).read_bytes()

    edit, tune = a / "brewery-weight-edit", a / "brewery-weight-tune"
    expectations = {edit: ["weight_inspect", "weight_propose", "weight_apply"], tune: ["weight_tune"]}
    for path, jobs in expectations.items():
        status = json.loads(command([path, "--capabilities"]))
        assert status == {"schema": "brewery.weight-runtime.v1", "jobs": jobs,
                          "formats": ["gguf", "hgn", "safetensors", "hf"], "visual": True,
                          "packed": True, "proof_source_fds": True}

    work = root / ".weight-gguf-smoke"
    work.mkdir()
    source = work / "source.gguf"
    original = matrix((1, 2, 3, 4))
    source.write_bytes(original)
    digest = checksum(source)
    tile = ["tile", source.name, "--source-sha256", digest, f"--tensor={TENSOR}",
            "--origin=[0,0]", "--shape=[2,2]"]
    assert json.loads(command([edit, *tile], cwd=work))["values"] == [[1, 2], [3, 4]]
    command([edit, *tile], cwd=root, ok=False)

    save(work / "calibration.json", {"activations": [[1, 0], [0, 1]], "targets": [[2, 3], [2, 3]]})
    private = root / "hf-fixture-credential"
    # This is synthetic, valid calibration data so the helper could use it
    # if an outside host path were incorrectly mounted.
    heldout = {"activations": [[1, 1], [2, -1]], "targets": [[3, 7], [0, 2]]}
    save(private, heldout)
    held = work / "held-out.json"
    held.symlink_to(private)
    tune_args = [tune, source.name, "--source-sha256", digest, f"--tensor={TENSOR}",
                 "--indices=[[0,0],[1,1]]", "--calibration", "calibration.json",
                 "--held-out", held.name, "--max-update", "1"]
    env = {**ENV, "HF_TOKEN": "synthetic-not-a-token", "HF_HOME": str(root), "PYTHONPATH": str(root)}
    command(tune_args, cwd=work, env=env, ok=False)
    held.unlink()
    save(held, heldout)
    proposal = json.loads(command(tune_args, cwd=work, env=env))
    assert [(e["index"], e["after"]) for e in proposal["edits"]] == [([0, 0], 2), ([1, 1], 3)]
    assert source.read_bytes() == original
    pinned_source_smoke(edit, root)

    tampered = b / "brewery-weight-runtime/weight_edit.py"
    tampered.chmod(0o600)
    tampered.write_text("raise RuntimeError('untrusted helper loaded')\n")
    command([b / edit.name, "--capabilities"], ok=False)
    confined_source_smoke(a, b, root)
    print(json.dumps({"dependency_sha256": pin, "manifest_sha256": manifest,
                      "reproducible": True, "tamper_rejected": True, "outside_source_rejected": True,
                      "readonly_pinned_member": True, "pinned_apply_no_clobber": True,
                      "hy4_confined_edit": True}))


if __name__ == "__main__":
    main()
