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
import time

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


def probe_export(root, worker, policy, source, assets):
    source = canonical_path(str(source))
    assets = canonical_path(str(assets))
    require(source.suffix == ".safetensors" and source.is_file() and source.stat().st_nlink == 1
            and 0 < source.stat().st_size <= 32 << 20 and metadata_bytes(assets) <= 32 << 20,
            "probe requires a tiny standalone SafeTensors fixture with bounded metadata")
    binary = root / "bin"
    wrapper = worker["admission"]
    before = {path: file_sha(path) for path in (source, *assets.iterdir())}
    with tempfile.TemporaryDirectory(prefix="probe-", dir=root) as name:
        base = Path(name)
        imports, state = base / "imports", base / "worker"
        imports.mkdir()
        state.mkdir()
        weights, metadata = imports / source.name, imports / "metadata"
        metadata.mkdir()
        shutil.copyfile(source, weights)
        expected_assets = {p.name: before[p] for p in before if p != source}
        for leaf in expected_assets:
            shutil.copyfile(assets / leaf, metadata / leaf, follow_symlinks=False)
        require(metadata_bytes(metadata) <= 32 << 20
                and {p.name: file_sha(p) for p in metadata.iterdir()} == expected_assets
                and file_sha(weights) == before[source],
                "standalone probe fixture changed while copying")
        # Use the exact staged binaries and admission wrapper, but never
        # register probe inputs or jobs in the destination's worker roots.
        roots = {"imports": str(imports), "state": str(state)}
        (root / "local-roots.json").write_text(json.dumps(roots) + "\n")
        (root / "standalone.json").write_text(json.dumps(policy | roots, default=str) + "\n")
        config = base / "worker.json"
        config.write_text(json.dumps(worker | {"root": str(state), "source_roots": [str(imports)]}) + "\n")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(base), "TMPDIR": str(base),
               "BREWERY_WORKER_CONFIG": str(config)}

        def rpc(op, **args):
            with (root / "standalone-probe.log").open("ab") as log:
                run = subprocess.run([wrapper, "rpc"], input=json.dumps({"op": op, **args}).encode(),
                                     stdout=subprocess.PIPE, stderr=log, env=env, timeout=15)
            require(run.returncode == 0 and run.stdout, f"standalone worker {op} probe exited {run.returncode}")
            response = json.loads(run.stdout)
            require("ok" in response, f"standalone worker {op} probe failed: {response.get('error')}")
            reply = response["ok"]
            if op != "status":
                require(not reply.get("error"), f"standalone worker {op} probe failed: {reply.get('error')}")
            return reply

        def wait(id, seconds):
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                reply = rpc("status", id=id)
                if reply["state"] not in {"queued", "running"}:
                    return reply
                time.sleep(0.2)
            raise ValueError(f"standalone worker {id} probe timed out")

        def conversion(kind, inputs):
            id = f"standalone-probe-{kind}"
            job = {"kind": kind, "options": {"output_format": "gguf", "gguf_type": "BF16", "threads": 2},
                   "inputs": [{"path": str(path), "kind": role, "host": HOST} for role, path in inputs]}
            require(rpc("prepare", id=id, action="start", job=job)["state"] == "queued",
                    f"standalone worker did not prepare {kind}")
            for i, item in enumerate(job["inputs"]):
                reply = rpc("resolve", id=id, index=i, input=item)
                require(all(reply[key] == item[key] for key in ("host", "path", "kind")),
                        "standalone worker substituted a probe input")
            try:
                rpc("start", id=id)
                reply = wait(id, 150)
            except (ValueError, OSError, subprocess.TimeoutExpired):
                rpc("cancel", id=id)
                wait(id, 10)
                raise
            if reply["state"] != "succeeded":
                error = reply.get("error", reply["state"])
                stderr = state / "jobs" / id / "engine-stderr.log"
                if stderr.is_file():
                    with stderr.open("rb") as log:
                        log.seek(max(0, stderr.stat().st_size - 2048))
                        error += ": " + log.read(2048).decode(errors="replace").strip()
                raise ValueError(f"standalone worker {kind} probe failed: {error}")
            report = reply["result"]
            backend = report["backend_receipt"]
            require(report["kind"] == kind and report["output_format"] == "gguf" and report["gguf_type"] == "BF16"
                    and report["file_verification"]["status"] == "passed"
                    and report["report"]["source_format"] == "safetensors"
                    and report["report"]["tensor_payloads_read"] is True
                    and report["report"]["hf_export"]["llama_cpp_commit"] == REVISION
                    and backend["executable_sha256"] == QUANT_SHA
                    and backend["hf_exporter_sha256"] == (root / "exporter.sha256").read_text().strip()
                    and [(row["kind"], row["path"]) for row in report["inputs"]]
                    == [(role, str(path)) for role, path in inputs],
                    f"invalid standalone worker {kind} probe receipt")
            require(report["inputs"][0]["sha256"] == before[source]
                    and {f["path"]: f["sha256"] for f in report["inputs"][-1]["files"]} == expected_assets,
                    f"standalone worker {kind} probed different source bytes")
            return report

        convert = conversion("convert", [("safetensors", weights), ("assets", metadata)])
        model = state / "jobs/standalone-probe-convert/work/converted.gguf"
        artifact = convert["artifact"]
        info = model.lstat()
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 128 << 20
                and artifact["kind"] == "gguf" and artifact["path"] == str(model)
                and artifact["bytes"] == info.st_size and artifact["sha256"] == file_sha(model),
                "invalid standalone worker probe output")
        with model.open("rb") as stream:
            require(stream.read(4) == b"GGUF", "standalone worker probe did not produce GGUF")
        verify = conversion("verify", [("safetensors", weights), ("gguf", model), ("assets", metadata)])
        require(verify["source_sha256"] == convert["source_sha256"]
                and verify["inputs"][1]["sha256"] == artifact["sha256"]
                and file_sha(model) == artifact["sha256"]
                and file_sha(weights) == before[source] and metadata_bytes(metadata) <= 32 << 20
                and {p.name: file_sha(p) for p in metadata.iterdir()} == expected_assets
                and {p: file_sha(p) for p in (source, *assets.iterdir())} == before,
                "standalone worker probe inputs or output changed")
        report = {"convert": convert, "verify": verify,
                  "binary_sha256": {n: file_sha(binary / n) for n in ("brewery-worker", "brewery-engine")}}
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
        local_roots = dict(imports=imports, state=state)
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
            policy = dict(runtime=runtime, llama=llama)
            worker["engine_env"] = {**worker.get("engine_env", {}),
                                    "BREWERY_GGUF_SAFETENSORS": "1",
                                    "BREWERY_GGUF_QUANTIZER": str(binary / "llama-quantize"),
                                    "BREWERY_GGUF_HF_EXPORTER": str(exporter)}
            probe_export(root, worker, policy, probe_source, probe_assets)
            (root / "standalone.json").write_text(json.dumps(policy | local_roots, default=str) + "\n")
            caps["gguf_safetensors_output"] = True
        if existing or runtime:
            (root / "local-roots.json").write_text(json.dumps(local_roots, default=str) + "\n")
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
