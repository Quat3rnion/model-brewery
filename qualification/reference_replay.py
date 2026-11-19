"""Run the derived-F32 experiment locally. Run under GPU admission and build locks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def run(args):
    spec = json.loads(Path(__file__).with_name("replay-inputs.json").read_text())
    pin = next(m for m in spec["models"] if m["kind"] == args.kind)
    source = args.source.resolve(strict=True)
    require(source.stat().st_size == pin["bytes"] and sha(source) == pin["sha256"],
            "source does not match the pinned model")
    tool = args.converter.resolve(strict=True)
    lf = args.lanternfish.resolve(strict=True)
    cmp = args.comparator.resolve(strict=True)
    size_probe = subprocess.check_output(
        [tool, "convert", source, "--size-only", pin["sha256"]], text=True)
    estimated = int(size_probe.split("gguf_bytes=", 1)[1].strip())
    require(shutil.disk_usage(args.out.parent).free > estimated + 1_073_741_824,
            "insufficient scratch storage")
    if args.gpu_layers is None:
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15).splitlines()
        require(len(memory) == 1, "this full-offload diagnostic requires a single visible NVIDIA GPU")
        require(int(memory[0]) * (1 << 20) > estimated + 1_073_741_824,
                f"full F32 reference needs {estimated} bytes of GGUF plus 1 GiB headroom; "
                f"only {memory[0]} MiB VRAM free")
    args.out.mkdir(mode=0o700, exist_ok=False)
    out = args.out.resolve()
    derived = out / "derived-f32.gguf"
    with (out / "convert.log").open("xb") as log:
        subprocess.run([tool, "convert", source, derived, pin["sha256"]],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    digest = sha(derived)
    require(digest != pin["sha256"] and sha(source) == pin["sha256"],
            "source changed or conversion did not produce a distinct GGUF")
    flags = {"NVIDIA_TF32_OVERRIDE": "0", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "fp32",
             "GGML_CUDA_DISABLE_FUSION": "1", "CUDA_VISIBLE_DEVICES": "0"}
    env = os.environ | flags
    diagnostic = {
        "schema": "brewery.reference-replay.v1",
        "scope": "derived-f32 diagnostic only: not native GGUF qualification",
        "source": pin,
        "derived": {"name": derived.name, "sha256": digest, "bytes": derived.stat().st_size},
        "converter_sha256": sha(tool),
        "lanternfish_sha256": sha(lf),
        "comparator_sha256": sha(cmp),
        "gpu_device": args.device,
        "gpu_layers": args.gpu_layers,
        "placement": "all layers requested" if args.gpu_layers is None else "mixed CPU/CUDA",
        "environment_overrides": flags,
        "replays": {},
    }
    cmd = [lf, "--verbose", "replay", derived, "--threads", "8",
           "--prefill", json.dumps(spec["prefill"]), "--decode", json.dumps(spec["decode"])]
    # The native-format payloads exist only in private scratch during the comparison.
    # Persist them inside a different outer schema that is not accepted as native evidence.
    with tempfile.TemporaryDirectory(dir=out, prefix=".compare-") as temp:
        paths = []
        for mode in ("cpu", "gpu"):
            path = Path(temp) / (mode + ".json")
            with path.open("xb") as result, (out / (mode + ".log")).open("xb") as log:
                offload = ["--gpu-device", args.device] if mode == "gpu" else []
                if mode == "gpu" and args.gpu_layers is not None:
                    offload += ["--gpu-layers", str(args.gpu_layers)]
                subprocess.run([*cmd, *offload], stdout=result, stderr=log, check=True, env=env)
            receipt = json.loads(path.read_bytes())
            require(receipt["model"]["sha256"] == digest
                    and receipt["model"]["name"].endswith(" (derived F32 reference)")
                    and receipt["prefill_tokens"] == spec["prefill"]
                    and receipt["decode_tokens"] == spec["decode"], "wrong replay identity")
            seq = receipt["execution"]["steps"]
            require(receipt["execution"]["backend"] == mode
                    and all((s["gpu_matmul_nodes"] > 0) == (mode == "gpu") for s in seq),
                    "wrong replay backend")
            if mode == "gpu":
                require(receipt["execution"]["device"]["name"] == args.device, "wrong GPU")
                require(args.gpu_layers is not None or all(s["cpu_matmul_nodes"] == 0 for s in seq),
                        "full reference offload had CPU matmuls")
            diagnostic["replays"][mode] = receipt
            paths.append(path)
        opt = ["--require-moe"] if args.kind == "moe" else []
        result = subprocess.run([cmp, *opt, *paths], capture_output=True, text=True)
        diagnostic["unchanged_comparator"] = {
            "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        }
        with (out / "reference-result.json").open("x") as f:
            json.dump(diagnostic, f, separators=(",", ":"))
            f.write("\n")
        return result.returncode


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("dense", "moe"))
    for opt in ("source", "out", "converter", "lanternfish", "comparator"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--device", choices=("CUDA0",), required=True)
    p.add_argument("--gpu-layers", type=int, help="explicit mixed-device diagnostic; default requests all")
    args = p.parse_args()
    if args.gpu_layers is not None and args.gpu_layers <= 0:
        p.error("--gpu-layers must be positive")
    raise SystemExit(run(args))
