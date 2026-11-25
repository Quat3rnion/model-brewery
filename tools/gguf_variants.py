#!/usr/bin/env python3
"""Make named GGUF variants with the Model Brewery CPU conversion engine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_variants(path):
    variants = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(variants, dict) or not variants:
        raise ValueError("expected a non-empty JSON object of named variants")
    for name, options in variants.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError(f"invalid variant name: {name!r}")
        if (not isinstance(options, dict) or "gguf_type" not in options
                or options.keys() - {"gguf_type", "tensor_types", "threads"}):
            raise ValueError(f"{name}: expected gguf_type and optional tensor_types and threads")
    return variants


def regular_file(path):
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return path


def identity(receipt):
    return {key: receipt[key] for key in ("path", "sha256", "bytes")}


def run(args):
    if not re.fullmatch(r"[a-f0-9]{64}", args.quantizer_sha256):
        raise ValueError("--quantizer-sha256 must be a lowercase SHA-256 digest")
    variants = read_variants(args.variants)
    engine = regular_file(args.engine)
    quantizer = regular_file(args.quantizer)
    inputs = [{"kind": "gguf", "host": "local", "path": str(regular_file(args.source))}]
    if args.imatrix:
        inputs.append({"kind": "imatrix", "host": "local", "path": str(regular_file(args.imatrix))})

    out = args.out.parent.resolve(strict=True) / args.out.name
    out.mkdir(mode=0o700)
    results = {}
    manifest = {
        "schema": "brewery.gguf-variants.v1",
        "release_qualified": False,
        "quantizer": {"path": str(quantizer), "sha256": args.quantizer_sha256},
        "variants": results,
    }
    with tempfile.TemporaryDirectory(prefix=".work-", dir=out) as directory:
        work = Path(directory)
        # Use the approved bytes in private storage, not the original pathname.
        staged = work / ".llama-quantize"
        shutil.copyfile(quantizer, staged)
        with staged.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != args.quantizer_sha256:
                raise ValueError("quantizer SHA-256 mismatch")
        staged.chmod(0o500)
        env = os.environ | {"BREWERY_GGUF_QUANTIZER": str(staged)}

        for name, options in variants.items():
            cwd = work / name
            cwd.mkdir()
            options = options | {"output_format": "gguf"}
            if args.imatrix:
                options["imatrix"] = 1
            job = {"kind": "convert", "inputs": inputs, "options": options}
            process = subprocess.run([engine], cwd=cwd, env=env, input=json.dumps(job),
                                     stdout=subprocess.PIPE, encoding="utf-8")
            if process.returncode:
                raise ValueError(f"{name}: brewery-engine exited with {process.returncode}")
            receipt = json.loads(process.stdout)
            artifact = receipt["artifact"]
            report = receipt["report"]
            verification = receipt["file_verification"]
            backend = receipt["backend_receipt"]
            if (verification["status"] != "passed" or report["passed"] is not True
                    or receipt["release_qualified"] is not False
                    or backend["executable_sha256"] != args.quantizer_sha256
                    or backend["requested_options"] != options
                    or artifact["kind"] != "gguf"
                    or artifact["path"] != str(cwd / "converted.gguf")):
                raise ValueError(f"{name}: invalid GGUF conversion receipt")
            # Pin the physical bytes of all inputs across variants.
            for item, verified in zip(inputs, receipt["inputs"], strict=True):
                item["sha256"] = verified["sha256"]
            manifest["source"] = identity(receipt["inputs"][0])
            if args.imatrix:
                manifest["imatrix"] = identity(receipt["inputs"][1])
            results[name] = {
                "artifact": identity(artifact) | {"path": f"{name}.gguf"},
                "gguf_type": report["gguf_type"],
                "threads": backend["threads"],
                "requested_tensor_types": report["requested_tensor_types"],
                "file_verification": verification,
            }

        pending = work / "manifest.json"
        pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in results:
            (work / name / "converted.gguf").rename(out / f"{name}.gguf")
        pending.rename(out / "manifest.json")
    return out / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="brewery-engine executable")
    parser.add_argument("--quantizer", type=Path, required=True, help="approved CPU llama-quantize")
    parser.add_argument("--quantizer-sha256", required=True, help="approved executable digest")
    parser.add_argument("--source", type=Path, required=True, help="single unquantized GGUF")
    parser.add_argument("--variants", type=Path, required=True, help="JSON variant definitions")
    parser.add_argument("--imatrix", type=Path, help="importance matrix for all variants")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    try:
        print(run(args))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"gguf_variants: {exc}\n")


if __name__ == "__main__":
    main()
