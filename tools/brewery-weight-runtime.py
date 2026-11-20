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
STOCK_LAUNCHERS = ("brewery-weight-edit", "brewery-weight-tune")
CONFINED_EDITOR = "brewery-weight-edit-confined.py"
LAUNCHERS = (*STOCK_LAUNCHERS, CONFINED_EDITOR)
MODULES = ("weight_edit.py", "weight_tune.py", "weight_safetensors.py", "weight_hgn.py",
           "weight_q8g64.py", "weight_ggml_codes.py", "cross_container/ggml_in_hgn.py")

# Embedded in both entrypoints; not imported through an unverified Python path.
# The stock shell launcher embeds this in a single-quoted shell string.
VERIFY_RUNTIME = """\
def verify_runtime(root, pin):
    receipt = root / "manifest.json"
    if not receipt.is_file() or receipt.is_symlink():
        sys.exit("invalid weight runtime manifest")
    payload = receipt.read_bytes()
    if hashlib.sha256(payload).hexdigest() != pin:
        sys.exit("weight runtime manifest pin does not match")
    manifest = json.loads(payload)
    if manifest["abi"] != sys.implementation.cache_tag:
        sys.exit("weight runtime Python ABI changed; rebuild the release")
    files = manifest["files"]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            sys.exit("symlink in weight runtime")
        if not path.is_dir():
            actual.add(str(path.relative_to(root)))
    if actual != set(files) | {"manifest.json"}:
        sys.exit("weight runtime file set does not match")
    for name, digest in files.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            sys.exit("nonregular file in weight runtime")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                sys.exit("weight runtime file does not match manifest: " + name)
    sys.path.insert(0, str(root))
"""

# Only for the pre-confined Hy4 worker. The worker service owns the namespace
# and its mount allowlist; this is not an alternative to running that sandbox.
CONFINED_SCRIPT = """\
#!/usr/bin/python3 -ISB
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

@VERIFY_RUNTIME@

def require(ok, message):
    if not ok:
        sys.exit(message)


def readonly(path):
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def main():
    entry = Path(__file__).resolve(strict=True)
    root = entry.with_name("brewery-weight-runtime")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    links = [line.split(":", 1)[0].strip()
             for line in Path("/proc/self/net/dev").read_text().splitlines()[2:]]
    require(readonly("/") and readonly("/proc") and readonly(entry) and readonly(root)
            and not root.is_symlink() and status.get("NoNewPrivs", "").strip() == "1"
            and all(int(status.get(field, "1").strip(), 16) == 0
                    for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))
            and links == ["lo"],
            "confined weight editor requires the restricted worker sandbox")
    args = sys.argv[1:]
    probe = args == ["--capabilities"]
    work = Path.cwd()
    os.environ.clear()
    os.environ.update(LC_ALL="C", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if not probe:
        check_job(work, args)
        os.environ.update(HOME=str(work), TMPDIR=str(work))
    verify_runtime(root, "@MANIFEST_SHA256@")
    import weight_edit
    if probe:
        print(json.dumps(dict(schema="brewery.weight-runtime.v1",
                              jobs=["weight_inspect", "weight_propose", "weight_apply"],
                              formats=["safetensors", "hf"], visual=True, packed=False,
                              proof_source_fds=False)))
        return 0
    sys.argv = [weight_edit.__file__, *args]
    return weight_edit.main()


def check_job(work, args):
    error = "confined weight editor accepts only verified SafeTensors scratch jobs"
    info = work.stat()
    require(work.name.startswith(".weight-safetensors-") and not readonly(work)
            and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
            error)
    require(len(args) >= 4 and args[0] in ("inspect", "inventory", "tile", "propose", "apply"), error)
    require(args[-2:] == ["--source-manifest", str(work / "source-manifest.json")], error)
    # Do not let abbreviated/duplicate argparse aliases substitute the proof.
    require(not any(arg.startswith("-") and "--source-manifest".startswith(arg.split("=")[0])
                    for arg in args[:-2]), error)

    def input_path(value, name, directory=False):
        path = work / name
        require(value in (name, str(path)) and not path.is_symlink()
                and (path.is_dir() if directory else path.is_file()), error)

    source = Path(args[1]).name
    require(source in ("source.hf", "source.safetensors"), error)
    input_path(args[1], source, directory=source == "source.hf")
    input_path(args[-1], "source-manifest.json")
    if args[0] == "propose":
        input_path(args[2], "request.json")
    elif args[0] == "apply":
        input_path(args[2], "proposal.json")
        name = "output.hf" if source == "source.hf" else "output.safetensors"
        output = work / name
        require(args[3] in (name, str(output)) and not os.path.lexists(output), error)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        sys.exit("confined weight runtime: " + str(exc))
"""


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
        launcher = (TOOLS / "brewery-weight-runtime-launch.sh").read_text()
        for name in LAUNCHERS:
            entry = scratch / name
            script = CONFINED_SCRIPT if name == CONFINED_EDITOR else launcher
            entry.write_text(script.replace("@MANIFEST_SHA256@", pin)
                             .replace("@VERIFY_RUNTIME@", VERIFY_RUNTIME))
            entry.chmod(0o500)
        for name in STOCK_LAUNCHERS:
            # Verify the actual confined runtime before publishing either launcher.
            subprocess.run([scratch / name, "--capabilities"], cwd=scratch, env=ENV,
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
