#!/usr/bin/python3
"""Stage local-gguf without editing active dispatch or worker configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

from local_gguf import (KINDS, ROCMFPX_QUANTIZER_ENV, ROCMFPX_QUANTIZER_NAME, ROCMFPX_REVISION,
                        ROCMFPX_SHA_ENV, canonical_path, metadata_bytes, pinned_native, require, sandbox)

HOST = "local-gguf"
REVISION = "184f6536024668e8908ab8c86033d56708e62b40"
QUANT_SHA = "7ceb0e39e4f7e207fe8c55215368163bb5734072dc871a66b45527e1a5c9c672"
# Separate CPU build from charlie12345/ROCmFPX at ROCMFPX_REVISION.
ROCMFPX_QUANT_SHA = "593e59cb7bcdd51ca4f9e7cac3ea42797c48dd1c18ac9b13552dd8c382ae3708"
ROCMFPX_PROBE_MAX = 8 << 20
DISABLED = {"baseline", "benchmark", "cache_cleanup", "compare", "evaluate", "generate",
            "transfer", "transfer_preview", "convert_components", "custom_profiles", "gguf_hf_output"}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trusted_dir(path):
    path = canonical_path(str(path))
    info = path.stat()
    require(path.is_dir() and info.st_uid in {0, os.getuid()} and not info.st_mode & 0o022,
            "expected a trusted canonical directory")
    return path


def probe_export(root, config, source, assets):
    source = canonical_path(str(source))
    assets = canonical_path(str(assets))
    require(source.suffix == ".safetensors" and source.is_file() and source.stat().st_nlink == 1
            and 0 < source.stat().st_size <= 32 << 20 and metadata_bytes(assets) <= 32 << 20,
            "probe requires a tiny standalone SafeTensors fixture with bounded metadata")
    binary = root / "bin"
    exporter = binary / "gguf-export-hf"
    with tempfile.TemporaryDirectory(prefix="probe-", dir=root) as name:
        base = Path(name)
        work = base / "work"
        work.mkdir()
        runtime_check = [str(config["runtime"] / "bin/python"), "-IB", "-c",
                         "import torch,transformers; assert torch.__version__=='2.11.0+cpu' "
                         "and torch.version.cuda is None and torch.version.hip is None "
                         "and transformers.__version__=='4.57.6'"]
        args, env = sandbox(work, [], runtime_check, config, binary)
        subprocess.run(args, env=env, check=True, capture_output=True, timeout=30)
        output = work / "converted"
        command = [str(exporter), "export", "--source", str(source), "--assets", str(assets),
                   "--outtype", "bf16", "--output", str(output)]
        args, env = sandbox(work, [source, assets], command, config, binary)
        with (root / "standalone-probe.log").open("wb") as log:
            result = subprocess.run(args, env=env, capture_output=True, timeout=90)
            log.write(result.stderr)
        require(result.returncode == 0, f"pinned adapter probe failed: {result.stderr[-2048:].decode(errors='replace')}")
        report = json.loads(result.stdout)
        model = output / "model.gguf"
        artifact = report["artifact"]
        require(report["source_format"] == "safetensors" and report["export_runtime_checked"] is True
                and report["tensor_payloads_read"] is True and report["llama_cpp_commit"] == REVISION
                and report["gguf_inventory"] and artifact["path"] == str(model)
                and artifact["bytes"] == model.stat().st_size and file_sha(model) == artifact["sha256"],
                "invalid standalone adapter probe receipt")
        with model.open("rb") as stream:
            require(stream.read(4) == b"GGUF", "standalone probe did not export GGUF")
        (root / "standalone-probe.json").write_text(json.dumps(report, indent=2) + "\n")


def probe_rocmfpx(root, source, sha):
    source = canonical_path(str(source))
    info = source.stat()
    require(source.suffix == ".gguf" and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and 0 < info.st_size <= ROCMFPX_PROBE_MAX,
            "ROCmFPX probe requires a tiny standalone unquantized GGUF fixture (maximum 8 MiB)")
    tensor = "blk.0.attn_q.weight"
    private_type = "Q4_0_ROCMFP4_FAST"
    options = {"output_format": "gguf", "gguf_type": "Q8_0", "threads": 1,
               "tensor_types": {tensor: private_type}, "encoder_family": "rocmfpx",
               "encoder_revision": ROCMFPX_REVISION}
    binary = root / "bin"
    with tempfile.TemporaryDirectory(prefix="rocmfpx-probe-", dir=root) as name:
        work = Path(name) / "convert/work"
        output = work / "converted.gguf"
        reports = {}
        for kind, paths in (("convert", [source]), ("verify", [source, output])):
            work = Path(name) / kind / "work"
            work.mkdir(parents=True)
            job = {"kind": kind, "options": options,
                   "inputs": [{"path": str(path), "kind": "gguf", "host": HOST} for path in paths]}
            cmd = ["/usr/bin/prlimit", f"--fsize={ROCMFPX_PROBE_MAX}", "--", str(binary / "brewery-engine")]
            args, env = sandbox(work, paths, cmd, binary=binary, fork_sha=sha)
            with tempfile.TemporaryFile(dir=work.parent) as stdout, tempfile.TemporaryFile(dir=work.parent) as stderr:
                result = subprocess.run(args, env=env, input=json.dumps(job).encode(), timeout=90,
                                        stdout=stdout, stderr=stderr)
                stderr.seek(max(0, stderr.tell() - 2048))
                require(result.returncode == 0, f"ROCmFPX {kind} probe failed: {stderr.read().decode(errors='replace')}")
                stdout.seek(0)
                report = json.load(stdout)
            backend = report["backend_receipt"]
            require(report["kind"] == kind and report["file_verification"]["status"] == "passed"
                    and report["report"]["tensor_payloads_read"] is True
                    and backend["executable_sha256"] == sha and backend["encoder_family"] == "rocmfpx"
                    and backend["encoder_revision"] == ROCMFPX_REVISION
                    and backend["stock_runtime_compatible"] is False
                    and any(row["name"] == tensor and row["current_type"].upper() == private_type
                            for row in report["report"]["gguf_tensors"]),
                    f"invalid ROCmFPX {kind} probe receipt")
            reports[kind] = report
            if kind == "convert":
                artifact = report["artifact"]
                output_info = output.lstat()
                require(stat.S_ISREG(output_info.st_mode) and 0 < output_info.st_size <= ROCMFPX_PROBE_MAX
                        and artifact["path"] == str(output) and artifact["bytes"] == output_info.st_size
                        and artifact["sha256"] == file_sha(output), "invalid ROCmFPX probe output")
        (root / "rocmfpx-probe.json").write_text(json.dumps(reports, indent=2) + "\n")


def stage(root, release, quantizer, catalog, dispatch_path, *, runtime=None, llama=None,
          probe_source=None, probe_assets=None, rocmfpx_quantizer=None, rocmfpx_sha256=None,
          rocmfpx_probe_source=None):
    root = root.absolute()
    if root.parent.resolve(strict=True) != root.parent:
        raise ValueError("expected canonical destination parent")
    require(file_sha(quantizer) == QUANT_SHA, "quantizer is not the verified CPU build")
    optional = (runtime, llama, probe_source, probe_assets)
    require(not any(optional) or all(optional), "standalone requires runtime, llama checkout, and both probe inputs")
    fork = (rocmfpx_quantizer, rocmfpx_sha256, rocmfpx_probe_source)
    require(not any(value is not None for value in fork) or all(fork),
            "ROCmFPX quantizer, SHA-256, and probe source must be supplied together")
    if rocmfpx_quantizer:
        rocmfpx_quantizer = canonical_path(str(rocmfpx_quantizer))
        info = rocmfpx_quantizer.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_mode & 0o111 and not rocmfpx_quantizer.samefile(quantizer)
                and rocmfpx_sha256 == ROCMFPX_QUANT_SHA and file_sha(rocmfpx_quantizer) == rocmfpx_sha256,
                "ROCmFPX quantizer is not the separately approved CPU build")
    dispatch = json.loads(dispatch_path.read_bytes())
    existing = dispatch["hosts"].get(HOST)
    old_config = {}
    if existing is not None:
        require(existing.get("local") is True
                and not any(existing.get(k) for k in ("ssh", "managed_ssh"))
                and existing.get("capabilities", {}).get("gguf_output") is True,
                "expected an existing direct local-gguf worker")
        require(runtime is not None or not existing["capabilities"].get("gguf_safetensors_output"),
                "an existing standalone upgrade requires its runtime, llama checkout, and both probe inputs")
        old = canonical_path(existing["config"])
        info = old.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid in {0, os.getuid()} and not info.st_mode & 0o022
                and info.st_size <= 64 << 10, "existing local-gguf config is not trusted")
        old_config = json.loads(old.read_bytes())
        require(len(old_config["source_roots"]) == 1 and "pinned_publication" not in old_config
                and "hf_download" not in old_config
                and Path(old_config["admission"]).name == "local_gguf.py",
                "expected an existing admitted direct local-gguf import root")
    catalog_values = json.loads(catalog.read_bytes())
    if catalog_values["encoder_revision"] != REVISION:
        raise ValueError("GGUF catalog has a different encoder revision")
    if rocmfpx_quantizer:
        require(catalog_values.get("rocmfpx_encoder_revision") == ROCMFPX_REVISION,
                "GGUF catalog has a different ROCmFPX encoder revision")
    os.umask(0o077)
    root.mkdir(mode=0o700)
    try:
        binary = root / "bin"
        imports = trusted_dir(Path(old_config["source_roots"][0])) if existing else root / "imports"
        state = trusted_dir(Path(old_config["root"])) if existing else root / "worker"
        for path in (binary,) if existing else (binary, imports, state):
            path.mkdir()
        here = Path(__file__).resolve().parent
        for src, name in ((release / "brewery-engine", "brewery-engine"),
                          (release / "brewery-worker", "brewery-worker"),
                          (quantizer, "llama-quantize"),
                          (here / "local_gguf.py", "local_gguf.py"),
                          (here / "local_components.py", "local_components.py")):
            dest = binary / name
            shutil.copyfile(src, dest)
            dest.chmod(0o700)
        shutil.copyfile(catalog, root / "gguf_catalog.json")
        (root / "quantizer.sha256").write_text(QUANT_SHA + "\n")
        wrapper = str(binary / "local_gguf.py")
        config = root / "worker.json"
        worker = {**old_config, "root": str(state), "source_roots": [str(imports)],
                  "engine": str(binary / "brewery-engine"), "admission": wrapper,
                  "disk_reserve_bytes": max(16 << 30, old_config.get("disk_reserve_bytes", 0))}
        caps = (dict(existing.get("capabilities", {})) if existing else
                dict.fromkeys(KINDS | {"gguf_output"}, True) | dict.fromkeys(DISABLED, False))
        caps.pop("gguf_rocmfpx_output", None)
        if "engine_env" in worker:
            worker["engine_env"] = {key: value for key, value in worker["engine_env"].items()
                                    if key not in (ROCMFPX_QUANTIZER_ENV, ROCMFPX_SHA_ENV)}
        if existing and any(caps.get(k) for k in ("weight_inspect", "weight_propose", "weight_tune", "weight_apply")):
            old_wrapper = canonical_path(existing["worker"])
            require(old_config["admission"] == str(old_wrapper),
                    "expected the existing local-gguf admission wrapper")
            packed = trusted_dir(old_wrapper.parent / "weight-runtime-packed")
            shutil.copytree(packed, binary / packed.name, symlinks=True)
        if existing and not runtime:
            (root / "local-roots.json").write_text(json.dumps(dict(imports=imports, state=state), default=str) + "\n")
        if runtime:
            runtime, llama = trusted_dir(runtime), trusted_dir(llama)
            require(len(runtime.parts) >= 3 and len(llama.parts) >= 3
                    and not runtime.is_relative_to(llama) and not llama.is_relative_to(runtime)
                    and not any(a.is_relative_to(b) or b.is_relative_to(a)
                                for a in (runtime, llama) for b in (imports, state, root)),
                    "runtime and pinned llama checkout must be dedicated trusted directories")
            interpreter = runtime / "bin/python"
            require(interpreter.resolve(strict=True).is_relative_to(runtime)
                    and interpreter.is_file(), "runtime must contain its Python interpreter")
            exporter = binary / "gguf-export-hf"
            shutil.copyfile(here.parents[1] / "tools/gguf-export-hf", exporter)
            exporter.chmod(0o700)
            (root / "exporter.sha256").write_text(file_sha(exporter) + "\n")
            policy = dict(imports=imports, state=state, runtime=runtime, llama=llama)
            probe_export(root, policy, probe_source, probe_assets)
            (root / "standalone.json").write_text(json.dumps(policy, default=str) + "\n")
            worker["engine_env"] = {**worker.get("engine_env", {}),
                                    "BREWERY_GGUF_SAFETENSORS": "1",
                                    "BREWERY_GGUF_QUANTIZER": str(binary / "llama-quantize"),
                                    "BREWERY_GGUF_HF_EXPORTER": str(exporter)}
            # The proposal is emitted only after the actual pinned exporter
            # succeeds through the same sandbox used for admitted jobs.
            caps["gguf_safetensors_output"] = True
        if rocmfpx_quantizer:
            staged_fork = binary / ROCMFPX_QUANTIZER_NAME
            shutil.copyfile(rocmfpx_quantizer, staged_fork)
            staged_fork.chmod(0o700)
            pin = root / "rocmfpx.sha256"
            pin.write_text(rocmfpx_sha256 + "\n")
            pinned_native(staged_fork, pin)
            probe_rocmfpx(root, rocmfpx_probe_source, rocmfpx_sha256)
            worker["engine_env"] = {**worker.get("engine_env", {}), ROCMFPX_QUANTIZER_ENV: str(staged_fork),
                                    ROCMFPX_SHA_ENV: rocmfpx_sha256}
            caps["gguf_rocmfpx_output"] = True
        config.write_text(json.dumps(worker, indent=2) + "\n")
        dispatch["hosts"][HOST] = {**(existing or {}), "local": True, "worker": wrapper,
                                    "config": str(config), "capabilities": caps}
        (root / "dispatch.proposed.json").write_text(json.dumps(dispatch, sort_keys=True, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(root)
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "release", "quantizer", "catalog", "dispatch"):
        cli.add_argument(f"--{field}", type=Path, required=True)
    cli.add_argument("--runtime", type=Path, help="dedicated self-contained CPU Python runtime")
    cli.add_argument("--llama", type=Path, help="trusted llama.cpp checkout at the exporter pin")
    cli.add_argument("--probe-source", type=Path, help="bounded known-good standalone SafeTensors fixture")
    cli.add_argument("--probe-assets", type=Path, help="matching fixture config and tokenizer directory")
    cli.add_argument("--rocmfpx-quantizer", type=Path, help="separately approved CPU-only ROCmFPX binary")
    cli.add_argument("--rocmfpx-sha256", help="approved ROCmFPX executable SHA-256")
    cli.add_argument("--rocmfpx-probe-source", type=Path, help="tiny unquantized ROCmFPX smoke fixture GGUF")
    args = cli.parse_args()
    stage(args.root, args.release, args.quantizer, args.catalog, args.dispatch,
           runtime=args.runtime, llama=args.llama, probe_source=args.probe_source, probe_assets=args.probe_assets,
           rocmfpx_quantizer=args.rocmfpx_quantizer, rocmfpx_sha256=args.rocmfpx_sha256,
           rocmfpx_probe_source=args.rocmfpx_probe_source)
