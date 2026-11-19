#!/usr/bin/python3
"""Build a private local-gguf enrollment proposal without editing dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from local_gguf import KINDS

HOST = "local-gguf"
REVISION = "184f6536024668e8908ab8c86033d56708e62b40"
QUANT_SHA = "7ceb0e39e4f7e207fe8c55215368163bb5734072dc871a66b45527e1a5c9c672"
DISABLED = {"baseline", "benchmark", "cache_cleanup", "compare", "evaluate", "generate",
            "transfer", "transfer_preview", "convert_components", "custom_profiles", "gguf_hf_output"}


def stage(root, release, quantizer, catalog, dispatch_path):
    root = root.absolute()
    if root.parent.resolve(strict=True) != root.parent:
        raise ValueError("expected canonical destination parent")
    with quantizer.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != QUANT_SHA:
            raise ValueError("quantizer is not the verified CPU build")
    dispatch = json.loads(dispatch_path.read_bytes())
    if HOST in dispatch["hosts"]:
        raise ValueError("local-gguf already exists")
    if json.loads(catalog.read_bytes())["encoder_revision"] != REVISION:
        raise ValueError("GGUF catalog has a different encoder revision")
    os.umask(0o077)
    root.mkdir(mode=0o700)
    try:
        binary = root / "bin"
        for path in (binary, root / "imports", root / "worker"):
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
        config.write_text(json.dumps({"root": str(root / "worker"), "source_roots": [str(root / "imports")],
                                      "engine": str(binary / "brewery-engine"), "admission": wrapper,
                                      "disk_reserve_bytes": 16 << 30}, indent=2) + "\n")
        dispatch["hosts"][HOST] = {"local": True, "worker": wrapper, "config": str(config),
                                    "capabilities": dict.fromkeys(KINDS | {"gguf_output"}, True)
                                    | dict.fromkeys(DISABLED, False)}
        (root / "dispatch.proposed.json").write_text(json.dumps(dispatch, sort_keys=True, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(root)
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "release", "quantizer", "catalog", "dispatch"):
        cli.add_argument(f"--{field}", type=Path, required=True)
    args = cli.parse_args()
    stage(args.root, args.release, args.quantizer, args.catalog, args.dispatch)
