#!/usr/bin/python3
"""Local-only CPU GGUF admission, independent of the HGN workers."""
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
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
ROCMFPX_REVISION = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c"
ROCMFPX_QUANTIZER_NAME = "llama-quantize-rocmfpx"
ROCMFPX_QUANTIZER = BIN / ROCMFPX_QUANTIZER_NAME
ROCMFPX_PIN = ROOT / "rocmfpx.sha256"
ROCMFPX_QUANTIZER_ENV = "BREWERY_GGUF_ROCMFPX_QUANTIZER"
ROCMFPX_SHA_ENV = "BREWERY_GGUF_ROCMFPX_SHA256"
EXPORTER = BIN / "gguf-export-hf"
EXPORTER_PIN = ROOT / "exporter.sha256"
STANDALONE = ROOT / "standalone.json"
LOCAL_ROOTS = ROOT / "local-roots.json"
WEIGHT = BIN / "weight-runtime-packed"
GIB = 1 << 30
LIMIT = 64 << 10
KINDS = {"inspect", "plan", "convert", "verify"}
WEIGHT_KINDS = {"weight_inspect", "weight_propose", "weight_tune", "weight_apply"}
WEIGHT_ENV = {"BREWERY_WEIGHT_EDIT": str(WEIGHT / "edit"),
              "BREWERY_WEIGHT_TUNE": str(WEIGHT / "tune"), "BREWERY_WEIGHT_VISUAL": "1",
              "BREWERY_WEIGHT_HGN": "1", "BREWERY_WEIGHT_PACKED": "1"}
MODEL_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}")
METADATA = set("config.json tokenizer.json tokenizer_config.json chat_template.jinja merges.txt "
               "generation_config.json vocab.json preprocessor_config.json video_preprocessor_config.json "
               "special_tokens_map.json added_tokens.json processor_config.json".split())


def metadata_bytes(path):
    require(path.is_dir(), "assets must be a flat local metadata directory")
    total = 0
    found = set()
    for member in path.iterdir():
        require(member.name in METADATA, "unsupported assets metadata filename")
        info = member.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "assets metadata must be regular files")
        found.add(member.name)
        total += info.st_size
        require(total <= 64 << 20, "assets metadata exceeds 64 MiB")
    require("config.json" in found, "assets metadata must include config.json")
    return total


def private_policy(path, fields):
    if not os.path.lexists(path):
        return None
    mode = path.lstat()
    require(stat.S_ISREG(mode.st_mode) and mode.st_uid == os.getuid() and mode.st_mode & 0o077 == 0
            and mode.st_size <= 8192, f"invalid private GGUF policy: {path.name}")
    config = json.loads(path.read_bytes())
    require(isinstance(config, dict) and config.keys() == fields, f"invalid GGUF policy fields: {path.name}")
    return config


def local_config():
    config = private_policy(LOCAL_ROOTS, {"imports", "state"})
    if not config:
        return None
    return {k: canonical_path(v) for k, v in config.items()}


def standalone_job(job):
    return job["kind"] in KINDS and job["inputs"] and job["inputs"][0]["kind"] == "safetensors"


def job_config(job):
    roots = local_config()
    if not standalone_job(job):
        return roots
    config = private_policy(STANDALONE, {"imports", "state", "runtime", "llama"})
    if config is None:
        return roots
    config = {k: canonical_path(v) for k, v in config.items()}
    require(config["imports"] == (roots["imports"] if roots else IMPORTS)
            and config["state"] == (roots["state"] if roots else STATE),
            "standalone GGUF policy does not match worker roots")
    interpreter = config["runtime"] / "bin/python"
    require(interpreter.is_file() and interpreter.resolve(strict=True).is_relative_to(config["runtime"]),
            "standalone runtime requires a Python interpreter inside its runtime")
    return config


def catalog():
    return json.loads((ROOT / "gguf_catalog.json").read_bytes())


def rocmfpx_pin(job):
    if job["kind"] not in KINDS or job["options"].get("encoder_family") != "rocmfpx":
        return None
    require(os.path.lexists(ROCMFPX_PIN) and os.path.lexists(ROCMFPX_QUANTIZER),
            "ROCmFPX quantizer is not staged")
    pinned_native(ROCMFPX_QUANTIZER, ROCMFPX_PIN)
    return ROCMFPX_PIN.read_text().strip()


def headroom(kind, source_bytes, matrix_bytes=0, state=None):
    available = ram_available()
    require(available >= 16 * GIB, f"local-gguf needs >=16 GiB available RAM (have {available} bytes)")
    # Unquantized input plus intermediate/output content, independent of the
    # requested preset and deliberately conservative. Native plan also gates output.
    needed = 16 * GIB + matrix_bytes + (source_bytes * 4 if kind in {"convert", "weight_apply"}
                                       else source_bytes if kind in WEIGHT_KINDS else 0)
    require(shutil.disk_usage(state or STATE).free >= needed, f"local-gguf needs {needed} bytes of free disk")


def published(path, filename, state=None):
    return (path.name == filename and path.parent.name == "work"
            and path.parent.parent.parent == (state or STATE) / "jobs")


def weight_inputs(kind, inputs, paths, imports, state):
    fmt = inputs[0]["kind"]
    extra = {"weight_tune": ("weight-calibration", "weight-calibration"),
             "weight_apply": ("weight-edit-proposal",)}.get(kind, ())
    require(fmt in {"gguf", "hgn"} and tuple(row["kind"] for row in inputs[1:]) == extra,
            "invalid weight job input kinds")
    source = paths[0]
    require(source.suffix == f".{fmt}"
            and (source.is_relative_to(imports) or published(source, f"edited.{fmt}", state)
                 or fmt == "gguf" and published(source, "converted.gguf", state))
            and 0 < source.stat().st_size <= 32 << 20, "invalid bounded local weight source")
    if kind == "weight_tune":
        require(all(p.is_relative_to(imports) and 0 < p.stat().st_size <= 8 << 20 for p in paths[1:]),
                "calibration and held-out inputs must be bounded local imports")
    if kind == "weight_apply":
        require(published(paths[1], "weight-proposal.json", state) and 0 < paths[1].stat().st_size <= 128 << 10,
                "weight apply requires a bounded worker-published proposal")
    return paths


def job_inputs(job, *, standalone=False, imports=None, state=None):
    imports, state = imports or IMPORTS, state or STATE
    kind = job["kind"]
    require(kind in KINDS | WEIGHT_KINDS and job.get("prompt") is None,
            "local-gguf only accepts GGUF conversion and weight jobs")
    options, inputs = job["options"], job["inputs"]
    require(isinstance(options, dict), "expected job options object")
    require(isinstance(inputs, list) and 1 <= len(inputs) <= 4
            and all(row["host"] == "local-gguf" for row in inputs),
            "use only local-gguf Library inputs; remote inputs are not supported")
    paths = [canonical_path(row["path"]) for row in inputs]
    require(len(set(paths)) == len(paths), "GGUF inputs must use distinct paths")
    if kind in WEIGHT_KINDS:
        require(all(p.is_file() for p in paths), "weight job inputs must be regular files")
        return weight_inputs(kind, inputs, paths, imports, state)
    require(isinstance(options, dict) and options.keys() <= {
        "output_format", "gguf_type", "tensor_types", "threads", "imatrix",
        "encoder_family", "encoder_revision",
    }, "unsupported GGUF option")
    require(options.get("output_format") == "gguf", "local-gguf requires GGUF output")
    threads = options.get("threads", 4)
    require(type(threads) is int and 1 <= threads <= 4, "local-gguf permits 1 to 4 CPU threads")
    asset_index = 2 if kind == "verify" else 1
    safe = inputs[0]["kind"] == "safetensors"
    require(not safe or standalone, "standalone SafeTensors GGUF is not staged")
    base = asset_index + int(safe)
    require(len(inputs) in (base, base + 1), "wrong number of GGUF conversion inputs")
    if safe:
        source, assets = paths[0], paths[asset_index]
        require(source.is_relative_to(imports) and source.suffix == ".safetensors" and source.is_file()
                and source.stat().st_nlink == 1 and 0 < source.stat().st_size <= 512 << 20,
                "source must be a bounded standalone SafeTensors import")
        require(inputs[asset_index]["kind"] == "assets" and assets.is_relative_to(imports) and assets != imports
                and not source.is_relative_to(assets), "assets must be registered separately under local imports")
        metadata_bytes(assets)
    else:
        require(inputs[0]["kind"] == "gguf" and paths[0].is_file()
                and (paths[0].is_relative_to(imports) or published(paths[0], "edited.gguf", state))
                and paths[0].suffix == ".gguf"
                and not re.search(r"-[0-9]{5}-of-[0-9]{5}\.gguf$", paths[0].name),
                "source must be a single GGUF file in the local-gguf import directory")
    if kind == "verify":
        out = paths[1]
        require(inputs[1]["kind"] == "gguf" and out.is_file() and published(out, "converted.gguf", state),
                "verify output must be a published local-gguf artifact")
    has_matrix = len(paths) == base + 1
    require(("imatrix" in options) == has_matrix
            and (not has_matrix or (type(options["imatrix"]) is int and options["imatrix"] == base
                  and inputs[base]["kind"] == "imatrix" and paths[base].is_file() and paths[base].is_relative_to(imports)
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
    selection = [presets[preset], *(types[value] for value in overrides.values())]
    fork = any(row.get("encoder_family") == "rocmfpx" for row in selection)
    if fork:
        require(options.get("encoder_family") == "rocmfpx", "ROCmFPX selection requires its encoder family")
        require(options.get("encoder_revision") == values.get("rocmfpx_encoder_revision") == ROCMFPX_REVISION,
                "ROCmFPX selection requires the pinned source revision")
        require(all(row.get("rocmfpx_supported", True) for row in selection),
                "GGUF selection is not supported by ROCmFPX")
    else:
        require("encoder_family" not in options and "encoder_revision" not in options,
                "stock GGUF options must not select a fork encoder")
    require(has_matrix or not any(row.get("requires_imatrix", False) for row in selection),
            "imatrix required for selected GGUF quantization")
    return paths


def check(job, config=None):
    paths = job_inputs(job, standalone=config is not None and "runtime" in config,
                       imports=config["imports"] if config else None, state=config["state"] if config else None)
    matrix_bytes = sum(p.stat().st_size for p in paths[1:]) if job["kind"] in WEIGHT_KINDS \
        else paths[-1].stat().st_size if "imatrix" in job["options"] else 0
    safe = job["inputs"][0]["kind"] == "safetensors"
    if safe:
        matrix_bytes += metadata_bytes(paths[2 if job["kind"] == "verify" else 1])
        if job["kind"] in {"convert", "verify"}:
            matrix_bytes += 128 << 20  # Adapter scratch allowance for GGUF metadata.
    headroom("convert" if safe and job["kind"] == "verify" else job["kind"], paths[0].stat().st_size, matrix_bytes,
             config["state"] if config else STATE)
    return paths


def precheck(req):
    op = req["op"]
    if op == "prepare":
        require(req["action"] == "start", "local-gguf does not stage transfers")
        check(req["job"], job_config(req["job"]))
        rocmfpx_pin(req["job"])
    else:
        require(op in {"resolve", "start", "status", "cancel", "finish", "suggest_paths"},
                "unapproved local-gguf worker operation")
        if op == "resolve":
            require(req.get("bundle") is None and req["input"]["host"] == "local-gguf",
                    "local-gguf requires direct local inputs")


def sandbox(work, paths, command, config=None, binary=None, *, fork_sha=None):
    binary = binary or BIN
    env = {"PATH": "/usr/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
           "BREWERY_GGUF_QUANTIZER": str(binary / "llama-quantize"), "BREWERY_CPU_THREADS": "4", "OMP_NUM_THREADS": "4",
           "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", **WEIGHT_ENV}
    if fork_sha:
        env.update({ROCMFPX_QUANTIZER_ENV: str(binary / ROCMFPX_QUANTIZER_NAME), ROCMFPX_SHA_ENV: fork_sha})
    shim = "/usr/bin/bwrap"
    args = [shim, "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-net",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--ro-bind", str(binary), str(binary)]
    if config:
        runtime, llama = config["runtime"], config["llama"]
        env.update(BREWERY_GGUF_HF_EXPORTER=str(binary / EXPORTER.name),
                   BREWERY_GGUF_PYTHON=str(runtime / "bin/python"), BREWERY_LLAMA_CPP=str(llama))
        for path in (runtime, llama):
            args.extend(("--ro-bind", str(path), str(path)))
    for path in paths:
        args.extend(("--ro-bind", str(path), str(path)))
    args.extend(("--bind", str(work.parent), str(work.parent), "--chdir", str(work), "--", *command))
    return args, env


def check_runtime(work, config):
    command = [str(config["runtime"] / "bin/python"), "-IB", "-c",
               "import torch, transformers; "
               "raise SystemExit(not (torch.__version__ == '2.11.0+cpu' "
               "and torch.version.cuda is None and torch.version.hip is None "
               "and transformers.__version__ == '4.57.6'))"]
    args, env = sandbox(work, [], command, config)
    try:
        result = subprocess.run(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError("standalone runtime version check timed out") from None
    require(result.returncode == 0, "standalone runtime requires CPU PyTorch 2.11.0+cpu and Transformers 4.57.6")


def admit():
    require(sys.argv[1:] == ["--wait", "--", str(ENGINE)], "unexpected local-gguf admission invocation")
    data = os.pread(0, LIMIT + 1, 0)
    require(len(data) <= LIMIT, "local-gguf engine request too large")
    job = json.loads(data)
    config = job_config(job)
    paths = check(job, config)
    pinned_native(QUANTIZER, PIN)
    fork_sha = rocmfpx_pin(job)
    native = config if standalone_job(job) else None
    if native:
        pinned_native(EXPORTER, EXPORTER_PIN)
    work = Path.cwd()
    require(work.name == "work" and work.parent.parent == (config["state"] if config else STATE) / "jobs",
            "unexpected local-gguf job directory")
    if native:
        check_runtime(work, native)
    args, env = sandbox(work, paths, [str(ENGINE)], native, fork_sha=fork_sha)
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
