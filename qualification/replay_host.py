"""Capture and compare the two pinned models on one worker under its GPU admission lock."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    spec = json.loads(Path(__file__).with_name("replay-inputs.json").read_text())
    for model in spec["models"]:
        path = getattr(args, model["kind"])
        if path.stat().st_size != model["bytes"]:
            raise ValueError(f"{model['kind']}: model size mismatch")
        if sha(path) != model["sha256"]:
            raise ValueError(f"{model['kind']}: model hash mismatch")
    # A run directory is single use. Keep partial traces for diagnosis, not as passes.
    args.out.mkdir(mode=0o700, exist_ok=False)
    lf = Path(shutil.which(args.lanternfish) or args.lanternfish).resolve(strict=True)
    compare = Path(shutil.which(args.comparator) or args.comparator).resolve(strict=True)
    inventory = {"hostname": platform.node(), "kernel": platform.release()}
    if shutil.which("nvidia-smi"):
        inventory["nvidia"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv"],
            text=True, timeout=15,
        )
    kfd = Path("/sys/class/kfd/kfd/topology/nodes")
    if kfd.exists():
        inventory["kfd"] = [p.read_text() for p in sorted(kfd.glob("*/properties"))]
    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    report = {"scope": "pairwise only; independently verify the selected device and driver",
              "lanternfish_sha256": sha(lf), "comparator_sha256": sha(compare), "pairs": {},
              "comparison_exit_codes": {}}
    for model in spec["models"]:
        kind = model["kind"]
        cmd = [lf, "replay", getattr(args, kind).resolve(), "--prefill", json.dumps(spec["prefill"]),
               "--decode", json.dumps(spec["decode"])]
        files = []
        for mode in ("cpu", "gpu"):
            path = args.out / f"{kind}.{mode}.json"
            with path.open("xb") as out, path.with_suffix(".stderr").open("xb") as err:
                device = ["--gpu-device", args.device, "--gpu-layers", "999"] if mode == "gpu" else []
                subprocess.run([*cmd, *device], stdout=out, stderr=err, check=True)
            trace = json.loads(path.read_bytes())
            if (trace["model"]["sha256"] != model["sha256"] or
                trace["prefill_tokens"] != spec["prefill"] or trace["decode_tokens"] != spec["decode"]):
                raise ValueError(f"{kind}.{mode}: trace input identity mismatch")
            files.append(path)
        stdout = args.out / f"{kind}.comparison"
        with stdout.open("xb") as output:
            rc = subprocess.run([compare, *(["--require-moe"] if kind == "moe" else []), *files],
                                stdout=output, stderr=subprocess.STDOUT).returncode
        report["comparison_exit_codes"][kind] = rc
        report["pairs"][kind] = {p.name: sha(p) for p in [*files, stdout]}
    failed = [kind for kind, rc in report["comparison_exit_codes"].items() if rc]
    partial = args.out / "result.partial"
    partial.write_text(json.dumps(report, indent=2))
    partial.rename(args.out / ("failed.json" if failed else "result.json"))
    if failed:
        raise ValueError("fixed-v1 comparison failed: " + ", ".join(failed))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, epilog="Run under the host GPU and quiet-build locks.")
    for name in ("dense", "moe", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--lanternfish", default="lanternfish")
    p.add_argument("--comparator", default="trace-parity")
    try:
        os.umask(0o077)
        run(p.parse_args())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
