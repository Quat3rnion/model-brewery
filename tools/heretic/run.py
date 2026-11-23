"""Prepare offline inputs, then call the unmodified Heretic CLI."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib

import tomli_w


COMMIT = "4f89482c0d63f2552a582215939910f9f45a3e5f"
MAX_FILE = 4 << 30
MAX_CONFIG = 128 << 10
RUNTIME_BUDGET = 20 << 30
PRESETS = {
    "default": "7c251932a17020997ce247e2c32c76162d194edd08e005a97ca9fa15852ae841",
    "noslop": "f58b61c2d7a09c0edd50f4b801411c732e8e9d48726573d22b57bbd2f29fd1a1",
    "nosyco": "98b6097c005185583446a45a74c6d946e545c3fd24960efcdd744f9d8815a4ad",
    "terse": "c740a6834ff97f64a89c4eccd4c54448b6161d14f6b747c62b6b418888419f4c",
}
ASSETS = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "tokenizer.model", "chat_template.jinja",
)
PLUGINS = {
    "scorers": {
        "heretic.scorers.keyword_rate.KeywordRate", "heretic.scorers.kl_divergence.KLDivergence",
        "heretic.scorers.response_length.ResponseLength",
    },
    "modifiers": {"heretic.modifiers.ara.ARA", "heretic.modifiers.abliteration.Abliteration"},
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def used_bytes(root):
    # uv may hard-link installed files to its cache; count each inode once.
    seen = set()
    total = 0
    for path in root.rglob("*"):
        st = path.lstat()
        key = (st.st_dev, st.st_ino)
        if key not in seen:
            total += st.st_blocks * 512
            seen.add(key)
    return total


def read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "expected a regular file")
        data = stream.read(limit + 1)
        need(len(data) <= limit, "input file exceeds its size limit")
        return data


def identity(root):
    # Content hash, not just dist-info/RECORD; sdists built by uv also count.
    env = hashlib.sha256()
    venv = root / "venv"
    paths = sorted(p for p in venv.rglob("*") if p.is_file() and
                   p.suffix != ".pyc" and p.name != ".lock")
    for path in paths:
        if path.is_symlink():
            need(path.parent == venv / "bin" and path.name.startswith("python"),
                 "unexpected virtual environment symlink")
        env.update(f"{path.relative_to(venv)}\0{sha(path)}\n".encode())
    need(version("heretic-llm") == "2.1.0", "Heretic runtime is not 2.1.0")
    return {
        "heretic": version("heretic-llm"),
        "commit": COMMIT,
        "licence": "AGPL-3.0-or-later",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "lock_sha256": sha(root / "uv.lock"),
        "environment_sha256": env.hexdigest(),
        "launcher_sha256": sha(Path(__file__)),
    }


def local_datasets(node, root, work=None, pins=None):
    if isinstance(node, list):
        return [local_datasets(item, root, work, pins) for item in node]
    if not isinstance(node, dict):
        return node
    if "dataset" not in node:
        return {key: local_datasets(value, root, work, pins) for key, value in node.items()}
    label = node["dataset"]
    digest = node.get("brewery_sha256")
    if digest is None:
        index = root / "prompts" / "index.json"
        mapping = json.loads(read(index, MAX_CONFIG)) if index.exists() else {}
        digest = mapping.get(label, {}).get("sha256")
    need(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
         f"no offline prompt pin for {label!r}")
    src = root / "prompts" / f"{digest}.txt"
    need(src.exists(), f"missing offline prompt file for {label!r}")
    data = read(src, 16 << 20)
    need(hashlib.sha256(data).hexdigest() == digest, "offline prompt SHA-256 mismatch")
    dest = work / f"{digest}.txt" if work else src
    if work:
        dest.write_bytes(data)
        pins[label] = digest
    result = {key: value for key, value in node.items()
              if key not in {"brewery_sha256", "commit", "config", "column"}}
    result["dataset"] = str(dest)
    if "split" in result:
        result["split"] = result["split"].removeprefix("train").removeprefix("test")
        need(re.fullmatch(r"\[\d*:\d*\]", result["split"]), "invalid offline prompt split")
    return result


def prepare(settings, source, work, seed, trial):
    for key, allow in PLUGINS.items():
        if key in settings:
            need(isinstance(settings[key], list) and 1 <= len(settings[key]) <= 8
                 and all(isinstance(item, dict) and item.get("plugin") in allow
                         for item in settings[key]), "unsupported Heretic plugin")
    effective = dict(settings)
    for key in ("evaluate_model", "collect_reproducibles", "reproduce", "model_commit",
                "upload_repo_id", "upload_repo_private", "upload_reproducibility_information",
                "upload_tags", "benchmark_tasks", "benchmark", "shell",
                "max_memory", "batch_size_test_prompts", "response_prefix_test_prompts"):
        effective.pop(key, None)
    # Heretic's probe defaults use upstream datasets; never invoke them.
    effective.update(model=str(source), model_action="save", export_strategy="merge",
                     study_checkpoint_dir=str(work / "checkpoints"),
                     checkpoint_action="restart", quantization="none", dtypes=["auto"],
                     device_map="cuda:0", seed=seed, trial_index=trial, max_shard_size="5GB")
    effective.setdefault("batch_size", 4)
    effective.setdefault("response_prefix", "")
    effective.setdefault("max_response_length", 100)
    need(type(effective["batch_size"]) is int and 1 <= effective["batch_size"] <= 32
         and isinstance(effective["response_prefix"], str), "offline probes must be disabled")
    need(type(effective["max_response_length"]) is int and 1 <= effective["max_response_length"] <= 512,
         "Heretic response limit must be between 1 and 512 tokens")
    return effective


def stage_source(src, source_assets, expected, destination):
    destination.mkdir()
    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        size = os.fstat(stream.fileno())
        need(stat.S_ISREG(size.st_mode) and 10 <= size.st_size <= MAX_FILE,
             "invalid standalone SafeTensors source")
        digest = hashlib.sha256()
        with (destination / "model.safetensors").open("xb") as out:
            copied = 0
            while block := stream.read(4 << 20):
                copied += len(block)
                need(copied <= size.st_size, "base grew during staging")
                out.write(block)
                digest.update(block)
        need(copied == size.st_size and digest.hexdigest() == expected,
             "Library base SafeTensors SHA-256 mismatch")
    assets = {}
    for name in ASSETS:
        path = source_assets / name
        if not path.exists():
            continue
        data = read(path, 32 << 20)
        if name.endswith(".json"):
            document = json.loads(data)
            need(isinstance(document, dict) and not ({"auto_map", "custom_pipelines"} & document.keys()),
                 "model assets must not reference remote code")
        (destination / name).write_bytes(data)
        assets[name] = hashlib.sha256(data).hexdigest()
    need("config.json" in assets and ({"tokenizer.json", "tokenizer.model"} & assets.keys()),
         "install standalone model config.json and tokenizer next to the Library SafeTensors file")
    return assets


def run(root, run_dir, source, options, *, cpu=False, assets=None):
    root = root.resolve(strict=True)
    size = source.stat().st_size
    need(10 <= size <= MAX_FILE and used_bytes(root) + 2 * size + (512 << 20) <= RUNTIME_BUDGET,
         "Heretic runtime and scratch exceed the 20 GiB budget")
    actual_identity = identity(root)
    need(json.loads(read(root / "identity.json", MAX_CONFIG)) == actual_identity,
         "pinned Heretic environment changed; run setup again")
    record = options["invocation"]
    kind = options["kind"]
    if kind == "heretic_reproduce":
        manifest = options["manifest"]
        need(manifest.get("version") == "4", "Heretic requires a version 4 community manifest")
        settings = manifest["settings"]
        need(settings.get("model_commit") == options.get("base_hf_commit"),
             "community manifest base Hugging Face commit mismatch")
        hashes = manifest["hashes"]
        need(isinstance(hashes, dict) and set(hashes) == {"model.safetensors"}
             and re.fullmatch("[0-9a-f]{64}", hashes.get("model.safetensors", "")),
             "community manifest must pin a single merged SafeTensors export")
        seed, trial = settings["seed"], settings.get("trial_index") or 0
        original = json.dumps(manifest, sort_keys=True)
    else:
        profile = record["profile"]
        need(re.fullmatch("[a-z][a-z0-9_-]{0,63}", profile), "invalid profile name")
        config = read(root / "configs" / f"{profile}.toml", MAX_CONFIG)
        if profile in PRESETS:
            need(hashlib.sha256(config).hexdigest() == PRESETS[profile], "preset SHA-256 mismatch")
        original = config.decode()
        settings = tomllib.loads(original)
        seed, trial = record["seed"], record["trial_index"]
    need(type(seed) is int and 0 <= seed < 2**32 and type(trial) is int and 0 <= trial < 128,
         "invalid Heretic seed or trial index")
    pins = {}
    with tempfile.TemporaryDirectory(dir=run_dir, prefix="private-") as temp:
        work = Path(temp)
        base = work / "base"
        assets = stage_source(source, assets or source.parent, options["base_sha256"], base)
        settings = local_datasets(settings, root, work, pins)
        effective = prepare(settings, base, work, seed, trial)
        # Plugin defaults can hide unpinned datasets. Require explicit prompt
        # tables for each built-in scorer/modifier actually used.
        names = [("scorer", row["plugin"].rsplit(".", 1)[-1])
                 for row in effective.get("scorers", [
                     {"plugin": "KeywordRate"}, {"plugin": "KLDivergence"}])]
        names += [("modifier", row["plugin"].rsplit(".", 1)[-1])
                  for row in effective.get("modifiers", [{"plugin": "ARA"}])]
        for group, name in names:
            plugin = effective.get(group, {}).get(name, {})
            keys = ("prompts",) if group == "scorer" else ("good_prompts", "bad_prompts")
            need(all(plugin.get(key) for key in keys), f"explicit offline {name} prompt tables required")
        if cpu:
            effective["device_map"] = "cpu"
        export = work / "export"
        config = work / "config.toml"
        effective["save_directory"] = str(export)
        args = [str(root / "venv/bin/python"), "-IB", str(root / "venv/bin/heretic")]
        if kind == "heretic_reproduce":
            path = work / "reproduce.json"
            path.write_text(json.dumps(dict(manifest, settings=effective)))
            config.write_text(tomli_w.dumps({"ignore_mismatches": False, "save_directory": str(export)}))
            args += ["--reproduce", str(path)]
        else:
            effective["n_trials"] = record["trials"]
            effective["n_startup_trials"] = min(record["trials"], 10)
            config.write_text(tomli_w.dumps(effective))
            args += ["--model", str(base)]
        native = Path(sys.base_prefix)
        interpreter = root / "venv/bin/python"
        if interpreter.is_symlink():
            target = interpreter.parent / os.readlink(interpreter)
            if not target.is_relative_to(native):
                native = target.parent.parent
        sandbox = ["/usr/bin/bwrap", "--unshare-net", "--unshare-ipc", "--die-with-parent",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
                   "--proc", "/proc", "--dev-bind", "/dev", "/dev", "--ro-bind", "/sys", "/sys",
                   "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
                   "--symlink", "usr/bin", "/bin", "--tmpfs", "/tmp",
                   "--ro-bind", str(native.resolve()), str(native),
                   "--ro-bind", str(root / "venv"), str(root / "venv"),
                   "--bind", str(work), str(work), "--chdir", str(work), "--"]
        env = {
            "PATH": f"{root}/venv/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(work),
            "TMPDIR": str(work), "HF_HOME": str(work / "hf"), "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "TERM": "dumb",
            "NO_COLOR": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "HERETIC_CONFIG_FILE": str(config),
        }
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        with (run_dir / "heretic.log").open("wb") as log:
            subprocess.run(sandbox + args, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
        if kind == "heretic":
            log = read(run_dir / "heretic.log", 16 << 20)
            match = re.search(rb"Restoring model from trial\s+(\d+)", log)
            need(match is not None, "Heretic did not report its selected trial")
            actual_trial = int(match[1])
        weights = export / "model.safetensors"
        need(weights.is_file() and not weights.is_symlink()
             and sorted(export.glob("*.safetensors")) == [weights]
             and weights.stat().st_size <= MAX_FILE, "Heretic did not export one merged SafeTensors file")
        exported = sha(weights)
        if kind == "heretic_reproduce":
            need(exported == hashes["model.safetensors"], "reproduced export SHA-256 mismatch")
        weights.rename(run_dir / "model.safetensors")
        journal = {
            "schema": "brewery.heretic-journal.v1", "kind": kind, "base_sha256": options["base_sha256"],
            "base_hf_commit": options.get("base_hf_commit"), "export_sha256": exported,
            "invocation": record, "seed": seed, "trial_index": trial, "assets": assets, "prompts": pins,
            "runtime": actual_identity, "effective_config": config.read_text(),
            "source_config" if kind == "heretic" else "community_manifest": original,
        }
        if kind == "heretic_reproduce":
            journal["effective_manifest"] = json.loads(path.read_text())
        else:
            journal["selected_study_trial"] = actual_trial
        (run_dir / "journal.json").write_text(json.dumps(journal, sort_keys=True))


if __name__ == "__main__":
    action, target, *args = sys.argv[1:]
    runtime = Path(target)
    if action == "identity" and not args:
        print(json.dumps(identity(runtime.resolve(strict=True)), sort_keys=True))
    elif action in ("run", "cpu-smoke") and len(args) in (3, 4):
        dest, model, request, *assets = map(Path, args)
        run(runtime, dest, model, json.loads(read(request, MAX_CONFIG)),
            cpu=action == "cpu-smoke", assets=next(iter(assets), None))
    else:
        raise SystemExit("invalid Heretic runner arguments")
