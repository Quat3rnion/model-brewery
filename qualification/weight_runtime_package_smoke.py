"""Verify a pinned, reproducible install and the real helper sandbox."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp

from managed_weight_smoke import TENSOR, checksum, matrix, save

REPO = Path(__file__).resolve().parents[1]
PACKAGER = ["/usr/bin/python3", "-IB", REPO / "tools/brewery-weight-runtime.py"]
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def command(args, *, cwd=None, env=ENV, ok=True):
    p = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=25)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stderr)
    return p.stdout.strip()


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
    for name in ("brewery-weight-edit", "brewery-weight-tune"):
        assert (a / name).read_bytes() == (b / name).read_bytes()

    edit, tune = a / "brewery-weight-edit", a / "brewery-weight-tune"
    expectations = {edit: ["weight_inspect", "weight_propose", "weight_apply"], tune: ["weight_tune"]}
    for path, jobs in expectations.items():
        status = json.loads(command([path, "--capabilities"]))
        assert status == {"schema": "brewery.weight-runtime.v1", "jobs": jobs,
                          "formats": ["gguf", "hgn", "safetensors", "hf"], "visual": True, "packed": True}

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

    tampered = b / "brewery-weight-runtime/weight_edit.py"
    tampered.chmod(0o600)
    tampered.write_text("raise RuntimeError('untrusted helper loaded')\n")
    command([b / edit.name, "--capabilities"], ok=False)
    print(json.dumps({"dependency_sha256": pin, "manifest_sha256": manifest,
                      "reproducible": True, "tamper_rejected": True, "outside_source_rejected": True}))


if __name__ == "__main__":
    main()
