#!/usr/bin/python3
"""Isolate the local CPU component worker; preview its config without installing."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import unicodedata

HOME = Path("/mnt/games/model-brewery")
OFFICIAL = HOME / "local-sm120"
ROOT = HOME / "local-components"
STATE = ROOT / "worker"
COMPONENTS = HOME / "components"
HOST = "local-components"
ROLES = {"assets", "vision", "ngram", "mtp"}
KINDS = {"inspect", "plan", "convert", "verify"}
BACKENDS = {"ht_backend", "i4r_backend", "q4cp_backend", "decode_backend"}
GIB = 1 << 30
MAX_INPUT = 64 << 10
WORKER = ROOT / "bin/brewery-worker"
ENGINE = ROOT / "bin/brewery-engine"
NATIVE = ROOT / "bin/bf16-to-hgn-components"
NATIVE_SHA = ROOT / "native.sha256"
CUSTOM = ROOT / "bin/bf16-to-hgn-custom"
CUSTOM_SHA = ROOT / "custom.sha256"
CATALOG = ROOT / "profile_catalog.json"
WRAPPER = ROOT / "bin/local_components.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pin():
    config = ROOT / "worker.json" if Path(__file__).resolve() == WRAPPER else OFFICIAL / "worker.json"
    return json.loads(config.read_bytes())["pinned_publication"]


def source_path():
    return Path(read_pin()["source"])


def configs(dispatch):
    require(HOST not in dispatch["hosts"], "component host is already configured")
    pin = read_pin()
    worker = {
        "root": str(STATE),
        "source_roots": [str(source_path()), str(COMPONENTS)],
        "pinned_publication": pin,
        "engine": str(ENGINE),
        "admission": str(WRAPPER),
        "engine_env": {
            "BREWERY_COMPONENT_CPU_ONLY": "1",
            "BREWERY_COMPONENT_CONVERTER": str(NATIVE),
            # This worker never accepts a job without a recipe or overlays.
            "BREWERY_CONVERTER": "/usr/bin/false",
            "BREWERY_CUSTOM_CONVERTER": str(CUSTOM),
            "BREWERY_CPU_THREADS": "4",
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
    }
    enabled = KINDS | {"convert_components", "custom_profiles"}
    disabled = {"baseline", "benchmark", "cache_cleanup", "compare", "evaluate", "generate",
                "transfer", "transfer_preview"}
    added = {**dispatch["hosts"], HOST: {
        "local": True,
        "worker": str(WRAPPER),
        "config": str(ROOT / "worker.json"),
        "component_cpu_only": True,
        "capabilities": dict.fromkeys(enabled, True) | dict.fromkeys(disabled, False),
    }}
    return worker, {**dispatch, "hosts": added}


def ram_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            size, unit = line.split()[1:]
            require(unit == "kB", "unexpected MemAvailable unit")
            return int(size) * 1024
    raise ValueError("cannot check available RAM")


def disk_available(root):
    return shutil.disk_usage(root).free


def headroom(job):
    available = ram_available()
    require(available >= 16 * GIB, f"local-components needs >=16 GiB available RAM (have {available} bytes)")
    # Use the existing standby's conservative 260 GB convert threshold. The
    # other stages write small reports and use the existing 16 GiB reserve.
    required = 16 * GIB
    if job["kind"] == "convert":
        # Account for overlays in addition to the base model. Count apparent
        # bytes so sparse inputs cannot disguise the conversion's output need.
        def regular_bytes(path):
            meta = path.lstat()
            require(stat.S_ISREG(meta.st_mode), "component files must be regular files")
            return meta.st_size

        def walk_error(error):
            raise error

        required = 260_000_000_000
        for item in job["inputs"][1:]:
            path = Path(item["path"])
            if path.is_file():
                required += regular_bytes(path)
                continue
            for root, dirs, files in os.walk(path, onerror=walk_error):
                require(all(stat.S_ISDIR((Path(root) / name).lstat().st_mode) for name in dirs),
                        "component directories must not contain symlinks")
                required += sum(regular_bytes(Path(root) / name) for name in files)
    require(disk_available(HOME) >= required, f"local-components needs {required} free bytes on {HOME}")


def canonical_path(value):
    path = Path(value)
    require(path.is_absolute() and str(path) == value and path == path.resolve(strict=True),
            "component job paths must be canonical and must not use symlinks")
    return path


def read_catalog():
    return json.loads(CATALOG.read_bytes())


def validate_recipe(recipe, kind, options):
    require(isinstance(recipe, dict) and set(recipe) == {
        "schema", "revision", "name", "catalog_sha256", "architecture", "base", "overrides",
    }, "expected a validated inline profile recipe, not a path")
    catalog = read_catalog()
    require(type(recipe["schema"]) is int and recipe["schema"] == 1
            and recipe["architecture"] == catalog["architecture"]
            and recipe["catalog_sha256"] == catalog["catalog_sha256"], "profile catalog mismatch")
    name, base, overrides = recipe["name"], recipe["base"], recipe["overrides"]
    require(isinstance(name, str) and name.strip() and len(name.encode()) <= 256
            and not any(unicodedata.category(c) == "Cc" for c in name), "invalid profile name")
    require(base in ("v2", "ht43")
            and (kind == "verify" or options.get("profile") == base), "recipe base mismatch")
    editable = {row["name"] for row in catalog["tensors"] if row["editable"] is True}
    alternate = "ht43" if base == "v2" else "v2"
    require(isinstance(overrides, dict) and 1 <= len(overrides) <= 4096
            and all(tensor in editable and len(tensor.encode()) <= 512 and choice == alternate
                    for tensor, choice in overrides.items()), "recipe contains an invalid tensor override")
    body = {key: value for key, value in recipe.items() if key != "revision"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    require(recipe["revision"] == hashlib.sha256(canonical).hexdigest(), "profile revision mismatch")


def validate_job(job):
    kind = job["kind"]
    require(kind in KINDS and job.get("prompt") is None, "local-components serves only conversion stages")
    inputs, options = job["inputs"], job.get("options", {})
    first_role = 2 if kind == "verify" else 1
    require(isinstance(options, dict), "job options must be an object")
    allowed = ROLES | {"profile_recipe"}
    if kind == "verify":
        allowed |= {"decode_backend"}
    else:
        allowed |= {"profile", "ht_backend", "i4r_backend", "q4cp_backend"}
        if kind != "inspect":
            allowed |= {"seed", "threads", "ht_beam"}
    require(options.keys() <= allowed, "unsupported component or recipe option")
    has_recipe = "profile_recipe" in options
    require(first_role + (not has_recipe) <= len(inputs) <= first_role + len(ROLES),
            "component jobs need the base and a saved recipe or role input")
    if has_recipe:
        validate_recipe(options["profile_recipe"], kind, options)
    require(all(item["host"] == HOST for item in inputs),
            "use the local-components Library alias and local component models; cross-host copies are disabled")
    for key in BACKENDS & options.keys():
        require(options[key] == "cpu", f"local-components is CPU-only; {key} must be cpu")
    source = canonical_path(inputs[0]["path"])
    require(inputs[0]["kind"] == "hf" and source == source_path() and source.is_dir(),
            "base input must use the pinned HF publication")
    if kind == "verify":
        out = canonical_path(inputs[1]["path"])
        require(inputs[1]["kind"] in {"hgn", "hgn-v2", "ht43"} and out.is_dir()
                and out.parent.parent.parent == STATE / "jobs"
                and out.parts[-2:] == ("work", "converted"),
                "verify output must be a local-components conversion artifact")
    seen = set()
    for role in ROLES & options.keys():
        index = options[role]
        require(type(index) is int and first_role <= index < len(inputs) and index not in seen,
                f"invalid index for component role {role}")
        seen.add(index)
        item = inputs[index]
        path = canonical_path(item["path"])
        require(item["kind"] == role and path.is_relative_to(COMPONENTS) and path != COMPONENTS
                and (path.is_dir() or role != "assets" and path.is_file()),
                f"{role} must be a matching input in the private components directory")
    require(seen == set(range(first_role, len(inputs))), "each component input must have one matching role")
    return kind


def precheck(req):
    op = req["op"]
    if op == "prepare":
        require(req["action"] == "start", "local-components permits only direct local conversion stages")
        validate_job(req["job"])
        headroom(req["job"])
    else:
        require(op in {"resolve", "start", "status", "cancel", "finish", "suggest_paths"},
                "local-components does not support transfers or unrelated work")
        if op == "resolve":
            require(req.get("bundle") is None and req["input"]["host"] == HOST,
                    "local-components requires direct local inputs")


def pinned_native(executable, pin):
    for path in (executable, pin):
        meta = path.lstat()
        require(stat.S_ISREG(meta.st_mode) and meta.st_uid == os.getuid() and meta.st_mode & 0o077 == 0,
                f"expected a private pinned component tool: {path}")
    expected = pin.read_text().strip()
    require(re.fullmatch(r"[a-f0-9]{64}", expected), "invalid component converter pin")
    with executable.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == expected, "component converter differs from the approved CPU build")


def admit():
    require(sys.argv[1:] == ["--wait", "--", str(ENGINE)], "unexpected component admission invocation")
    data = os.pread(0, MAX_INPUT + 1, 0)
    require(len(data) <= MAX_INPUT, "component engine request too large")
    job = json.loads(data)
    kind = validate_job(job)
    headroom(job)
    pinned_native(NATIVE, NATIVE_SHA)
    if "profile_recipe" in job["options"]:
        pinned_native(CUSTOM, CUSTOM_SHA)
        require(not CUSTOM.samefile(NATIVE), "custom profile converter must be distinct")
    work = Path.cwd()
    require(work.name == "work" and work.parent.parent == STATE / "jobs", "invalid component job directory")
    # A private device namespace contains no GPU; keep sources read-only and
    # permit writes only to this job's own directory.
    env = {"PATH": "/usr/bin", "LC_ALL": "C.UTF-8", "HOME": "/nonexistent",
           **configs({"hosts": {}})[0]["engine_env"]}
    sandbox = "/usr/bin/bwrap"
    args = [sandbox, "--die-with-parent", "--unshare-user", "--unshare-net", "--unshare-pid",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in (ROOT / "bin", source_path(), COMPONENTS):
        args.extend(("--ro-bind", str(path), str(path)))
    if kind == "verify":
        out = job["inputs"][1]["path"]
        args.extend(("--ro-bind", out, out))
    args.extend(("--bind", str(work.parent), str(work.parent), "--chdir", str(work), "--", str(ENGINE)))
    os.execve(sandbox, args, env)


def main():
    os.umask(0o077)
    if sys.argv[1:] == ["preview"]:
        worker, dispatch = configs(json.loads((OFFICIAL / "dispatch.json").read_bytes()))
        print(json.dumps({"worker": worker, "dispatch": dispatch}, indent=2, sort_keys=True))
    elif sys.argv[1:] == ["rpc"]:
        data = sys.stdin.buffer.read(MAX_INPUT + 1)
        require(len(data) <= MAX_INPUT, "component RPC request too large")
        precheck(json.loads(data))
        # Worker RPC reads through EOF; regular temp files avoid pipe deadlocks
        # and leave the child in the existing dispatch process hierarchy.
        fd = os.memfd_create("component-rpc")
        with os.fdopen(fd, "wb", closefd=False) as body:
            body.write(data)
        os.lseek(fd, 0, os.SEEK_SET)
        os.dup2(fd, 0, inheritable=True)
        os.close(fd)
        os.execv(WORKER, [str(WORKER), "rpc"])
    else:
        admit()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        if sys.argv[1:] == ["rpc"]:
            print(json.dumps({"error": str(error)}))
        else:
            print(f"Component admission denied: {error}", file=sys.stderr)
            sys.exit(1)
