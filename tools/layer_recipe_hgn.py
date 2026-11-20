#!/usr/bin/env python3
"""Export bounded dense Llama or Qwen2 checkpoints to structural native HGN v2 bundles."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

from weight_safetensors import (
    CHUNK, INDEX, OPEN_DIR, OPEN_FILE, Files, hash_fd, need, open_at, open_path,
    parse_json, read_at, rename_noreplace, stamp,
)


SCHEMA = "brewery.layer-quant.v1"
RECEIPT = "layer-recipe.receipt.json"
FLOAT_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
DEFAULT_TARGETS = (*FLOAT_BYTES, "Q8G64")
TRELLIS_BITS = {"HT3": 3, "HT4": 4}
TARGETS = (*DEFAULT_TARGETS, *TRELLIS_BITS)
ARCHITECTURES = {"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM"}
NATIVE_IDENTITY = "brewery-layer-recipe-structural-v1"
TRELLIS_SPEC_COMMIT = "79e75fd4ae7642478823525b0a3212951f584af5"
# bf16-to-hgn eaa15cb465410c95adf355f3208a34a95d6e81b9, as pinned in Cargo.lock.
TRELLIS_ENCODER_SHA256 = "e686f78a79315addc20b312235b835108fced8474844832fb50b882a7f7458ff"
ASSETS = {
    "config.json": 1 << 20, "tokenizer.json": 32 << 20,
    "tokenizer_config.json": 2 << 20, "generation_config.json": 1 << 20,
    "special_tokens_map.json": 1 << 20, "added_tokens.json": 2 << 20,
    "tokenizer.model": 8 << 20, "chat_template.jinja": 1 << 20,
    "vocab.json": 8 << 20, "merges.txt": 8 << 20,
    "README.md": 1 << 20, "LICENSE": 1 << 20, "LICENSE.txt": 1 << 20,
    ".gitattributes": 1 << 20,
}


def parse_recipe(path):
    with open_path(path, OPEN_FILE) as fd:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 1 << 20,
             "recipe must be a bounded regular file")
        raw = read_at(fd, 0, info.st_size)
        need(stamp(info) == stamp(os.fstat(fd)), "recipe changed during read")
    recipe = parse_json(raw)
    required = {"schema", "source_sha256", "layers"}
    need(isinstance(recipe, dict) and set(recipe) in (required, required | {"default_type"})
         and recipe["schema"] == SCHEMA
         and isinstance(recipe["source_sha256"], str)
         and re.fullmatch("[0-9a-f]{64}", recipe["source_sha256"]),
         "invalid brewery.layer-quant.v1 recipe or source SHA-256")
    layers = recipe["layers"]
    need(isinstance(layers, dict) and len(layers) <= 256
         and all(re.fullmatch("0|[1-9][0-9]{0,2}", layer) for layer in layers),
         "invalid layer IDs")
    for layer, target in layers.items():
        need(target in TARGETS,
             f"unsupported native HGN types: layer {layer} requests {target!r}; "
             f"supported types are {', '.join(TARGETS)} (HT43 is a profile, not a codec)")
    need("default_type" not in recipe or recipe["default_type"] in DEFAULT_TARGETS,
         "invalid default native HGN type; HT3 and HT4 are layer overrides only")
    return recipe, hashlib.sha256(raw).hexdigest()


@contextmanager
def asset_inputs(source, assets):
    if source.directory:
        need(assets is None, "indexed SafeTensors uses its source directory for assets")
        shards = {tensor.member for tensor in source.tensors.values()}
        names = set(source.names) - shards - {INDEX}
        need(not any(name.endswith(".safetensors") for name in names),
             "unreferenced SafeTensors shards in indexed source")
        need(set(source.dirs) == {""}, "nested asset directories are not supported")
        yield names, source.stat, source.open_member
        return
    need(assets is not None, "standalone SafeTensors requires --assets with config.json and tokenizer.json")
    with open_path(assets, OPEN_DIR) as root:
        initial = stamp(os.fstat(root))
        same_dir = Path(os.path.abspath(assets)) == Path(os.path.abspath(source.parent))
        files = {}
        with os.scandir(root) as entries:
            for entry in entries:
                if same_dir and entry.name == source.path.name:
                    continue
                need(entry.name in ASSETS, f"unsupported model asset: {entry.name}")
                files[entry.name] = entry.stat(follow_symlinks=False)

        @contextmanager
        def open_asset(name):
            with open_at(root, [name], OPEN_FILE) as fd:
                expected = stamp(files[name])
                current = os.fstat(fd)
                need(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                     and stamp(current) == expected, f"invalid or changed model asset: {name}")
                yield fd
                need(stamp(os.fstat(fd)) == expected, f"model asset changed: {name}")

        yield set(files), files, open_asset
        for name in files:
            with open_asset(name):
                pass
        with open_path(assets, OPEN_DIR) as current:
            need(initial == stamp(os.fstat(current)), "standalone asset directory changed")


def copy_assets(source, assets, work):
    hashes = {}
    with asset_inputs(source, assets) as (names, infos, opener):
        need({"config.json", "tokenizer.json"} <= names, "config.json and tokenizer.json are required")
        need(names <= ASSETS.keys(), f"unsupported model assets: {sorted(names - ASSETS.keys())}")
        for name in sorted(names):
            size = infos[name].st_size
            need(0 < size <= ASSETS[name], f"model asset exceeds size bounds: {name}")
            digest = hashlib.sha256()
            with opener(name) as fd, (work / name).open("xb") as out:
                for offset in range(0, size, CHUNK):
                    raw = read_at(fd, offset, min(CHUNK, size - offset))
                    digest.update(raw)
                    out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            hashes[name] = digest.hexdigest()
            if source.directory:
                need(hashes[name] == source.hashes[name], f"model asset SHA-256 mismatch: {name}")
    if "tokenizer_config.json" in hashes:
        tc = parse_json((work / "tokenizer_config.json").read_bytes())
        need(isinstance(tc, dict) and not tc.get("auto_map"), "custom tokenizers are not supported")
    return hashes


def check_tokenizer(path, vocab_size, model_label):
    tokenizer = parse_json(path.read_bytes())
    model = tokenizer.get("model") if isinstance(tokenizer, dict) else None
    need(isinstance(model, dict), "missing tokenizer model")
    vocab, kind = model.get("vocab"), model.get("type")
    if kind == "Unigram":
        need(isinstance(vocab, list) and vocab, "invalid Unigram tokenizer vocabulary")
        ids = range(len(vocab))
    else:
        need(kind in ("BPE", "WordLevel", "WordPiece") and isinstance(vocab, dict) and vocab,
             "unsupported or incomplete tokenizer vocabulary")
        ids = vocab.values()
    added = tokenizer.get("added_tokens", [])
    need(isinstance(added, list) and all(isinstance(row, dict) and "id" in row for row in added),
         "invalid tokenizer added_tokens")
    need(all(type(i) is int and 0 <= i < vocab_size for i in ids)
         and all(type(row["id"]) is int and 0 <= row["id"] < vocab_size for row in added),
         f"tokenizer vocabulary IDs exceed {model_label} embeddings")


def tensor_plan(tensors, recipe, config):
    model_type = config.get("model_type") if isinstance(config, dict) else None
    need(model_type in ARCHITECTURES and config.get("architectures") == [ARCHITECTURES[model_type]]
         and not any(config.get(k) for k in (
             "auto_map", "quantization_config", "vision_config", "text_config", "num_local_experts",
             "mlp_bias", "rope_scaling",
         )), "only dense LlamaForCausalLM or Qwen2ForCausalLM without custom code or RoPE scaling is supported")
    qwen2 = model_type == "qwen2"
    label = "Qwen2" if qwen2 else "Llama"
    need(not config.get("use_sliding_window") and config.get("attention_bias", qwen2) is qwen2,
         f"unsupported {label} attention configuration")

    def dim(key, limit):
        value = config.get(key)
        need(type(value) is int and 0 < value <= limit, f"invalid {label} config field: {key}")
        return value

    hidden = dim("hidden_size", 8192 if qwen2 else 65536)
    ffn = dim("intermediate_size", 32768 if qwen2 else 65536)
    heads = dim("num_attention_heads", min(hidden, 128) if qwen2 else hidden)
    kv_heads = dim("num_key_value_heads", heads) if qwen2 or "num_key_value_heads" in config else heads
    depth = dim("num_hidden_layers", 64 if qwen2 else 256)
    vocab = dim("vocab_size", 262144 if qwen2 else 1 << 20)
    dim("max_position_embeddings", 1 << 24)
    head_dim = hidden // heads
    need(hidden % heads == heads % kv_heads == head_dim % 2 == 0
         and config.get("head_dim", head_dim) == head_dim, f"unsupported {label} attention dimensions")
    tied = config.get("tie_word_embeddings", False)
    need(config.get("hidden_act", "silu") == "silu"
         and type(tied) is bool and not (qwen2 and tied), f"unsupported {label} topology")
    shapes = {"model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,)}
    if not tied or "lm_head.weight" in tensors:
        shapes["lm_head.weight"] = (vocab, hidden)
    assigned, default = recipe["layers"], recipe.get("default_type")
    need(all(int(layer) < depth for layer in assigned), f"layer ID outside {label} topology")
    if qwen2:
        need(default is not None, "Qwen2 recipe requires default_type for embeddings, head and unoverridden matrices")
    per_matrix = {"model.embed_tokens.weight": default, "lm_head.weight": default} if qwen2 else {}
    for layer in range(depth):
        target = assigned.get(str(layer), default)
        need(target is not None, f"missing layer assignment: {layer}")
        prefix = f"model.layers.{layer}."
        for suffix, shape in {
            "self_attn.q_proj": (hidden, hidden),
            "self_attn.k_proj": (kv_heads * head_dim, hidden),
            "self_attn.v_proj": (kv_heads * head_dim, hidden),
            "self_attn.o_proj": (hidden, hidden),
            "mlp.gate_proj": (ffn, hidden),
            "mlp.up_proj": (ffn, hidden),
            "mlp.down_proj": (hidden, ffn),
        }.items():
            name = prefix + suffix + ".weight"
            shapes[name], per_matrix[name] = shape, target
            if qwen2 and suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                shapes[prefix + suffix + ".bias"] = shape[:1]
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + norm + ".weight"] = (hidden,)
    missing, extra = shapes.keys() - tensors.keys(), tensors.keys() - shapes.keys()
    need(not missing and not extra,
         f"incomplete or unsupported {label} tensor topology: missing={sorted(missing)}, extra={sorted(extra)}")
    plan = []
    for name, t in sorted(tensors.items()):
        need(t.shape == shapes[name], f"incorrect {label} tensor shape: {name}")
        need(t.dtype in FLOAT_BYTES, f"unsupported source tensor dtype: {name} ({t.dtype})")
        if qwen2 and len(t.shape) == 1:
            need(t.dtype == "F32", f"{name}: Qwen2 recipe requires source F32 vectors (no silent widening)")
        target = per_matrix.get(name, t.dtype)
        if target == "Q8G64":
            need(t.shape[-1] % 64 == 0, f"Q8G64 probes require columns divisible by 64: {name}")
        elif target in TRELLIS_BITS:
            need(len(t.shape) == 2 and all(128 <= d <= 2**24 and d % 128 == 0 for d in t.shape),
                 f"{target} requires rows and columns divisible by 128: {name} {t.shape}")
        else:
            need(FLOAT_BYTES[target] <= FLOAT_BYTES[t.dtype], f"widening is not supported: {name}")
        plan.append({"name": name, "shape": t.shape, "dtype": t.dtype, "target": target})
    return plan


def write_input(source, plan, stream):
    stream.write(json.dumps(plan, separators=(",", ":")).encode("ascii") + b"\n")
    hashes = {}
    for row in plan:
        t = source.tensors[row["name"]]
        size = math.prod(t.shape) * FLOAT_BYTES[t.dtype]
        digest = hashlib.sha256()
        with source.open_member(t.member) as fd:
            for offset in range(0, size, CHUNK):
                raw = read_at(fd, t.start + offset, min(CHUNK, size - offset))
                digest.update(raw)
                stream.write(raw)
        hashes[t.name] = digest.hexdigest()
    stream.seek(0)
    return hashes


def helper_run(helper, command, model, **kwargs):
    out = subprocess.run([helper, command, str(model)], capture_output=True, timeout=120, **kwargs)
    need(out.returncode == 0,
         f"native HGN {command} failed: {out.stderr[:2048].decode('utf-8', errors='replace').strip()}")
    return out.stdout


def export(source_path, recipe_path, output, assets, helper):
    recipe, recipe_sha = parse_recipe(recipe_path)
    source = Files(source_path, recipe["source_sha256"])
    need(not os.path.lexists(output), "destination already exists")
    dest = Path(os.path.abspath(output))
    for protected in (source.path if source.directory else None, assets):
        need(protected is None or not dest.is_relative_to(Path(os.path.abspath(protected))),
             "destination cannot be inside the source or asset directory")
    with open_path(output.parent, OPEN_DIR):
        pass
    with tempfile.TemporaryDirectory(prefix=".layer-recipe-hgn-", dir=output.parent) as scratch:
        stage = Path(scratch) / "bundle"
        stage.mkdir(mode=0o700)
        assets_hashes = copy_assets(source, assets, stage)
        cfg = parse_json((stage / "config.json").read_bytes())
        plan = tensor_plan(source.tensors, recipe, cfg)
        label = "Qwen2" if cfg["model_type"] == "qwen2" else "Llama"
        check_tokenizer(stage / "tokenizer.json", cfg["vocab_size"], label)
        model = stage / "model.hgn"
        with tempfile.TemporaryFile(dir=scratch) as stream:
            source_hashes = write_input(source, plan, stream)
            if cfg.get("tie_word_embeddings") and "lm_head.weight" in source.tensors:
                head, embed = source.tensors["lm_head.weight"], source.tensors["model.embed_tokens.weight"]
                need(head.dtype == embed.dtype and source_hashes[head.name] == source_hashes[embed.name],
                     "tied Llama head differs from input embeddings")
            helper_run(helper, "write", model, stdin=stream)
        native = parse_json(helper_run(helper, "inspect", model))
        receipt = make_receipt(recipe, recipe_sha, assets_hashes, plan, source_hashes, native, model, cfg)
        with (stage / RECEIPT).open("x", encoding="utf-8") as out:
            json.dump(receipt, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        source.check_current()
        with open_path(stage, OPEN_DIR) as fd:
            os.fsync(fd)
        rename_noreplace(stage, output)
    return receipt


def make_receipt(recipe, recipe_sha, assets, plan, source_hashes, native, model, config):
    # Upstream Reader verifies all entries and payload checksums in the helper.
    need(isinstance(native, dict) and native.get("version") == 2
         and native.get("identity") == NATIVE_IDENTITY and isinstance(native.get("tensors"), list),
         "invalid native Reader report")
    rows = {row["name"]: row for row in native["tensors"]}
    need(len(rows) == len(native["tensors"]), "duplicate native HGN tensor name")
    trellis = any(row["target"] in TRELLIS_BITS for row in plan)
    if trellis:
        need(native.get("spec_commit") == TRELLIS_SPEC_COMMIT
             and native.get("encoder_sha256") == TRELLIS_ENCODER_SHA256,
             "native trellis helper does not match the pinned encoder and HGN spec; rebuild with Cargo.lock")
    details = []
    for expected in plan:
        name = expected["name"]
        found = rows.pop(name, None)
        target, src = expected["target"], expected["dtype"]
        need(found is not None and found["shape"] == list(expected["shape"])
             and found["dtype"] == target, f"native HGN tensor metadata mismatch: {name}")
        digest = found["sha256"]
        need(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
             f"invalid native HGN payload hash: {name}")
        original = source_hashes[name]
        need(src != target or digest == original, f"unchanged HGN payload differs: {name}")
        detail = {"name": name, "shape": expected["shape"], "source_type": src,
                  "native_type": target, "source_payload_sha256": original,
                  "native_payload_sha256": digest}
        if target in TRELLIS_BITS:
            bits = TRELLIS_BITS[target]
            n, k = expected["shape"]
            need(found["code"] == 16 and found["variant"] == 0x1200 + 2 * bits
                 and found["size"] == n * k * bits // 8,
                 f"native trellis ABI mismatch: {name}")
            rotation = {}
            for suffix, length in (("suh", k), ("svh", n)):
                companion = name.removesuffix(".weight") + "." + suffix
                plane = rows.pop(companion, None)
                need(plane is not None and plane["dtype"] == "F16" and plane["code"] == 2
                     and plane["variant"] == 0 and plane["shape"] == [length]
                     and plane["size"] == 2 * length
                     and isinstance(plane["sha256"], str) and re.fullmatch("[0-9a-f]{64}", plane["sha256"]),
                     f"missing or invalid native rotation plane: {companion}")
                rotation[suffix] = {"name": companion, "shape": [length], "native_type": "F16",
                                    "native_payload_sha256": plane["sha256"]}
            detail["rotation"] = rotation
        details.append(detail)
    need(not rows, "unexpected native HGN tensors")
    with open_path(model, OPEN_FILE) as fd:
        size = os.fstat(fd).st_size
        model_sha = hash_fd(fd, size)
    receipt = {
        "schema": "brewery.layer-quant-hgn.receipt.v1",
        "source_sha256": recipe["source_sha256"], "recipe_sha256": recipe_sha,
        "format": "native_hgn_v2", "architecture": config["architectures"][0],
        "structural_reader_verified": True,
        "runtime_load_verified": False,
        "runtime_blocker": f"Halogen native HgnSource does not support {config['architectures'][0]}",
        "model": {"path": "model.hgn", "bytes": size, "sha256": model_sha},
        "asset_sha256": assets, "tensors": details,
    }
    if trellis:
        receipt["trellis"] = {"spec_commit": TRELLIS_SPEC_COMMIT, "encoder_sha256": TRELLIS_ENCODER_SHA256}
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="standalone .safetensors or indexed HF directory")
    parser.add_argument("recipe", type=Path, help="source-bound brewery.layer-quant.v1 JSON")
    parser.add_argument("output", type=Path, help="new HGN bundle directory")
    parser.add_argument("--assets", type=Path, help="directory with standalone checkpoint config/tokenizer")
    parser.add_argument("--helper", required=True, help="trusted upstream native HGN adapter executable")
    args = parser.parse_args(argv)
    try:
        receipt = export(args.source, args.recipe, args.output, args.assets, args.helper)
    except (ValueError, OSError, KeyError, TypeError, RecursionError, subprocess.SubprocessError) as exc:
        print(f"layer-recipe-hgn: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"receipt": str(args.output / RECEIPT),
                      "model_sha256": receipt["model"]["sha256"],
                      "runtime_load_verified": receipt["runtime_load_verified"],
                      "runtime_blocker": receipt["runtime_blocker"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
