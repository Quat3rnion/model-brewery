#!/usr/bin/python3
"""Local-only CPU GGUF admission, independent of the HGN workers."""
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from local_components import canonical_path, pinned_native, ram_available, require

# The same enrollment layout supports scratch and live workers without sharing
# a state root. This file is executable only from an operator-owned bin dir.
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "worker"
IMPORTS = ROOT / "imports"
BIN = ROOT / "bin"
ENGINE = BIN / "brewery-engine"
WORKER = BIN / "brewery-worker"
QUANTIZER = BIN / "llama-quantize"
PIN = ROOT / "quantizer.sha256"
WEIGHT = BIN / "weight-runtime-packed"
GIB = 1 << 30
LIMIT = 64 << 10
KINDS = {"inspect", "plan", "convert", "verify"}
WEIGHT_KINDS = {"weight_inspect", "weight_propose", "weight_tune", "weight_apply"}
WEIGHT_ENV = {"BREWERY_WEIGHT_EDIT": str(WEIGHT / "edit"),
              "BREWERY_WEIGHT_TUNE": str(WEIGHT / "tune"), "BREWERY_WEIGHT_VISUAL": "1",
              "BREWERY_WEIGHT_HGN": "1", "BREWERY_WEIGHT_PACKED": "1"}
MODEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}")


def catalog():
    return json.loads((ROOT / "gguf_catalog.json").read_bytes())


def headroom(kind, source_bytes, matrix_bytes=0):
    available = ram_available()
    require(available >= 16 * GIB, f"local-gguf needs >=16 GiB available RAM (have {available} bytes)")
    # Unquantized input plus intermediate/output content, independent of the
    # requested preset and deliberately conservative. Native plan also gates output.
    needed = 16 * GIB + matrix_bytes + (source_bytes * 4 if kind in {"convert", "weight_apply"}
                                       else source_bytes if kind in WEIGHT_KINDS else 0)
    require(shutil.disk_usage(STATE).free >= needed, f"local-gguf needs {needed} bytes of free disk")


def published(path, filename):
    return (path.name == filename and path.parent.name == "work"
            and path.parent.parent.parent == STATE / "jobs")


def weight_inputs(kind, inputs, paths):
    fmt = inputs[0]["kind"]
    extra = {"weight_tune": ("weight-calibration", "weight-calibration"),
             "weight_apply": ("weight-edit-proposal",)}.get(kind, ())
    require(fmt in {"gguf", "hgn"} and tuple(row["kind"] for row in inputs[1:]) == extra,
            "invalid weight job input kinds")
    source = paths[0]
    require(source.suffix == f".{fmt}"
            and (source.is_relative_to(IMPORTS) or published(source, f"edited.{fmt}")
                 or fmt == "gguf" and published(source, "converted.gguf"))
            and 0 < source.stat().st_size <= 32 << 20, "invalid bounded local weight source")
    if kind == "weight_tune":
        require(all(p.is_relative_to(IMPORTS) and 0 < p.stat().st_size <= 8 << 20 for p in paths[1:]),
                "calibration and held-out inputs must be bounded local imports")
    if kind == "weight_apply":
        require(published(paths[1], "weight-proposal.json") and 0 < paths[1].stat().st_size <= 128 << 10,
                "weight apply requires a bounded worker-published proposal")
    return paths


def job_inputs(job):
    kind = job["kind"]
    require(kind in KINDS | WEIGHT_KINDS and job.get("prompt") is None,
            "local-gguf only accepts GGUF conversion and weight jobs")
    options, inputs = job["options"], job["inputs"]
    require(isinstance(options, dict), "expected job options object")
    require(isinstance(inputs, list) and 1 <= len(inputs) <= 3
            and all(row["host"] == "local-gguf" for row in inputs),
            "use only local-gguf Library inputs; remote inputs are not supported")
    paths = [canonical_path(row["path"]) for row in inputs]
    require(len(set(paths)) == len(paths) and all(p.is_file() for p in paths),
            "GGUF inputs must be distinct regular files")
    if kind in WEIGHT_KINDS:
        return weight_inputs(kind, inputs, paths)
    require(isinstance(options, dict) and options.keys() <= {
        "output_format", "gguf_type", "tensor_types", "threads", "imatrix",
    }, "unsupported GGUF option")
    require(options.get("output_format") == "gguf", "local-gguf requires GGUF output")
    threads = options.get("threads", 4)
    require(type(threads) is int and 1 <= threads <= 4, "local-gguf permits 1 to 4 CPU threads")
    base = 2 if kind == "verify" else 1
    require(len(inputs) in (base, base + 1), "wrong number of GGUF conversion inputs")
    require(inputs[0]["kind"] == "gguf" and (paths[0].is_relative_to(IMPORTS) or published(paths[0], "edited.gguf"))
            and paths[0].suffix == ".gguf"
            and not re.search(r"-[0-9]{5}-of-[0-9]{5}\.gguf$", paths[0].name),
            "source must be a single GGUF file in the local-gguf import directory")
    if kind == "verify":
        out = paths[1]
        require(inputs[1]["kind"] == "gguf" and out.name == "converted.gguf"
                and out.parent.name == "work" and out.parent.parent.parent == STATE / "jobs",
                "verify output must be a published local-gguf artifact")
    has_matrix = len(paths) == base + 1
    require(("imatrix" in options) == has_matrix
            and (not has_matrix or (type(options["imatrix"]) is int and options["imatrix"] == base
                  and inputs[base]["kind"] == "imatrix" and paths[base].is_relative_to(IMPORTS)
                  and 0 < paths[base].stat().st_size <= 256 << 20)),
            "imatrix must reference the final direct local import (maximum 256 MiB)")
    values = catalog()
    presets = {row["id"]: row for row in values["presets"]}
    types = {row["id"]: row for row in values["tensor_types"]}
    preset = options.get("gguf_type")
    overrides = options.get("tensor_types", {})
    require(isinstance(preset, str) and preset in presets, "unapproved GGUF preset")
    require(isinstance(overrides, dict) and len(overrides) <= 256
            and all(MODEL_NAME.fullmatch(name) and isinstance(value, str) and value in types
                    for name, value in overrides.items()), "unapproved GGUF tensor override")
    require(has_matrix or not any(row.get("requires_imatrix", False)
            for row in [presets[preset], *(types[value] for value in overrides.values())]),
            "imatrix required for selected GGUF quantization")
    return paths


def check(job):
    paths = job_inputs(job)
    matrix_bytes = sum(p.stat().st_size for p in paths[1:]) if job["kind"] in WEIGHT_KINDS \
        else paths[-1].stat().st_size if "imatrix" in job["options"] else 0
    headroom(job["kind"], paths[0].stat().st_size, matrix_bytes)
    return paths


def precheck(req):
    op = req["op"]
    if op == "prepare":
        require(req["action"] == "start", "local-gguf does not stage transfers")
        check(req["job"])
    else:
        require(op in {"resolve", "start", "status", "cancel", "finish", "suggest_paths"},
                "unapproved local-gguf worker operation")
        if op == "resolve":
            require(req.get("bundle") is None and req["input"]["host"] == "local-gguf",
                    "local-gguf requires direct local inputs")


def sandbox(work, paths, command):
    env = {"PATH": "/usr/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
           "BREWERY_GGUF_QUANTIZER": str(QUANTIZER), "BREWERY_CPU_THREADS": "4", "OMP_NUM_THREADS": "4",
           "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", **WEIGHT_ENV}
    shim = "/usr/bin/bwrap"
    args = [shim, "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-net",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--ro-bind", str(BIN), str(BIN)]
    for path in paths:
        args.extend(("--ro-bind", str(path), str(path)))
    args.extend(("--bind", str(work.parent), str(work.parent), "--chdir", str(work), "--", *command))
    return args, env


def admit():
    require(sys.argv[1:] == ["--wait", "--", str(ENGINE)], "unexpected local-gguf admission invocation")
    data = os.pread(0, LIMIT + 1, 0)
    require(len(data) <= LIMIT, "local-gguf engine request too large")
    job = json.loads(data)
    paths = check(job)
    pinned_native(QUANTIZER, PIN)
    work = Path.cwd()
    require(work.name == "work" and work.parent.parent == STATE / "jobs", "unexpected local-gguf job directory")
    args, env = sandbox(work, paths, [str(ENGINE)])
    os.execve(args[0], args, env)


def main():
    os.umask(0o077)
    if sys.argv[1:] != ["rpc"]:
        admit()
        return
    raw = sys.stdin.buffer.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "local-gguf worker request too large")
    precheck(json.loads(raw))
    fd = os.memfd_create("gguf-rpc")
    with os.fdopen(fd, "wb", closefd=False) as sink:
        sink.write(raw)
    os.lseek(fd, 0, os.SEEK_SET)
    os.dup2(fd, 0, inheritable=True)
    os.close(fd)
    os.execv(WORKER, [str(WORKER), "rpc"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        if sys.argv[1:] == ["rpc"]:
            print(json.dumps({"error": str(error)}))
        else:
            print(f"GGUF admission denied: {error}", file=sys.stderr)
            sys.exit(1)
