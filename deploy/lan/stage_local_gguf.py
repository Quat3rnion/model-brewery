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

from local_gguf import KINDS, canonical_path, metadata_bytes, require, sandbox

HOST = "local-gguf"
REVISION = "184f6536024668e8908ab8c86033d56708e62b40"
QUANT_SHA = "7ceb0e39e4f7e207fe8c55215368163bb5734072dc871a66b45527e1a5c9c672"
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


def stage(root, release, quantizer, catalog, dispatch_path, *, runtime=None, llama=None,
          probe_source=None, probe_assets=None):
    root = root.absolute()
    if root.parent.resolve(strict=True) != root.parent:
        raise ValueError("expected canonical destination parent")
    require(file_sha(quantizer) == QUANT_SHA, "quantizer is not the verified CPU build")
    optional = (runtime, llama, probe_source, probe_assets)
    require(not any(optional) or all(optional), "standalone requires runtime, llama checkout, and both probe inputs")
    dispatch = json.loads(dispatch_path.read_bytes())
    existing = dispatch["hosts"].get(HOST)
    old_config = {}
    if existing is not None:
        require(runtime is not None and existing.get("local") is True
                and not any(existing.get(k) for k in ("ssh", "managed_ssh"))
                and existing.get("capabilities", {}).get("gguf_output") is True,
                "local-gguf already exists; only standalone upgrades of the local worker can be staged")
        old = canonical_path(existing["config"])
        info = old.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid in {0, os.getuid()} and not info.st_mode & 0o022
                and info.st_size <= 64 << 10, "existing local-gguf config is not trusted")
        old_config = json.loads(old.read_bytes())
        require(len(old_config["source_roots"]) == 1 and "pinned_publication" not in old_config
                and "hf_download" not in old_config
                and Path(old_config["admission"]).name == "local_gguf.py",
                "expected an existing admitted direct local-gguf import root")
    if json.loads(catalog.read_bytes())["encoder_revision"] != REVISION:
        raise ValueError("GGUF catalog has a different encoder revision")
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
            if existing and any(caps.get(k) for k in ("weight_inspect", "weight_propose", "weight_tune", "weight_apply")):
                old_wrapper = canonical_path(existing["worker"])
                require(old_config["admission"] == str(old_wrapper),
                        "expected the existing local-gguf admission wrapper")
                packed = trusted_dir(old_wrapper.parent / "weight-runtime-packed")
                shutil.copytree(packed, binary / packed.name, symlinks=True)
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
    args = cli.parse_args()
    stage(args.root, args.release, args.quantizer, args.catalog, args.dispatch,
          runtime=args.runtime, llama=args.llama, probe_source=args.probe_source, probe_assets=args.probe_assets)
