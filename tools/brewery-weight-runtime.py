#!/usr/bin/python3 -I
"""Package the offline CPU weight runtime beside a fresh controller release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

TOOLS = Path(__file__).resolve().parent
PYTHON = "/usr/bin/python3"
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
ASSETS = "brewery-weight-runtime"
LAUNCHERS = ("brewery-weight-edit", "brewery-weight-tune")
MODULES = ("weight_edit.py", "weight_tune.py", "weight_safetensors.py", "weight_hgn.py",
           "weight_q8g64.py", "weight_ggml_codes.py", "cross_container/ggml_in_hgn.py")


def encode(value):
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"expected a regular dependency directory: {root}")
    entries = {}
    for item in sorted(root.rglob("*")):
        rel = item.relative_to(root)
        if "__pycache__" in rel.parts or item.suffix in {".pyc", ".pyo"}:
            continue
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(f"unsupported dependency entry: {item}")
        if item.is_file():
            entries[rel] = item
    if not entries:
        raise ValueError(f"no dependency files in {root}")
    return entries


def dependency_files(gguf_python):
    # Do not import third-party code to find its files, and ignore user site.
    code = """import importlib.util, json, sys
print(json.dumps([sys.implementation.cache_tag,
    {name: list(importlib.util.find_spec(name).submodule_search_locations)[0]
     for name in ("numpy", "yaml")}]))
"""
    abi, packages = json.loads(subprocess.check_output([PYTHON, "-IB", "-c", code], env=ENV))
    packages = {k: Path(v) for k, v in packages.items()}
    packages["gguf"] = gguf_python / "gguf"
    for name in ("numpy", "yaml"):
        libraries = packages[name].with_name(name + ".libs")
        if libraries.exists():
            packages[libraries.name] = libraries
    return abi, {str(Path(name) / rel): src for name, root in packages.items()
                 for rel, src in tree(root).items()}


def install(bin_dir, abi, dependencies, expected, supplied):
    if supplied != expected:
        raise ValueError("weight runtime dependency pin does not match; no files installed")
    bin_dir = bin_dir.resolve(strict=True)
    info = bin_dir.stat()
    if not bin_dir.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError("release directory must be owned by this user and not writable by others")
    paths = [bin_dir / name for name in (ASSETS, *LAUNCHERS)]
    if any(os.path.lexists(path) for path in paths):
        raise ValueError("install only into a fresh release directory; runtime already present")
    published = []
    with TemporaryDirectory(prefix=".weight-package-", dir=bin_dir) as tmp:
        scratch = Path(tmp)
        payload = scratch / ASSETS
        payload.mkdir()
        files = {**dependencies, **{name: TOOLS / name for name in MODULES}}
        for name, src in files.items():
            dest = payload / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            dest.chmod(0o400)
        copied = {name: digest(payload / name) for name in files}
        if hashlib.sha256(encode({"abi": abi, "files": {k: copied[k] for k in dependencies}})).hexdigest() != supplied:
            raise ValueError("weight runtime dependencies changed during packaging")
        manifest = payload / "manifest.json"
        manifest.write_bytes(encode({"abi": abi, "dependency_sha256": supplied, "files": copied}))
        manifest.chmod(0o400)
        pin = digest(manifest)
        launcher = (TOOLS / "brewery-weight-runtime-launch.sh").read_text().replace("@MANIFEST_SHA256@", pin)
        for name in LAUNCHERS:
            entry = scratch / name
            entry.write_text(launcher)
            entry.chmod(0o500)
            # Verify the actual confined runtime before publishing either launcher.
            subprocess.run([entry, "--capabilities"], cwd=scratch, env=ENV,
                           stdout=subprocess.DEVNULL, check=True, timeout=15)
        try:
            # Reserve the directory without clobbering an existing release.
            paths[0].mkdir(mode=0o700)
            published.append(paths[0])
            for item in sorted(payload.iterdir()):
                item.rename(paths[0] / item.name)
            for path in paths[1:]:
                os.link(scratch / path.name, path)
                published.append(path)
            for directory in [p for p in paths[0].rglob("*") if p.is_dir()] + [paths[0]]:
                directory.chmod(0o500)
        except BaseException:
            for path in reversed(published):
                if path == paths[0]:
                    for directory in [path, *(p for p in path.rglob("*") if p.is_dir())]:
                        directory.chmod(0o700)
                    shutil.rmtree(path)
                else:
                    path.unlink()
            raise
    print(pin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fingerprint", "install"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--gguf-python", type=Path, required=True, help="trusted llama.cpp gguf-py directory")
        if name == "install":
            cmd.add_argument("--bin-dir", type=Path, required=True)
            cmd.add_argument("--dependency-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        abi, dependencies = dependency_files(args.gguf_python.resolve(strict=True))
        fingerprint = hashlib.sha256(encode(
            {"abi": abi, "files": {name: digest(path) for name, path in dependencies.items()}}
        )).hexdigest()
        if args.command == "fingerprint":
            print(fingerprint)
        else:
            install(args.bin_dir, abi, dependencies, fingerprint, args.dependency_sha256)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"weight runtime: {exc}\n")


if __name__ == "__main__":
    main()
