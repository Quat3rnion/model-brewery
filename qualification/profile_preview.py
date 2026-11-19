#!/usr/bin/env python3
"""Preview whole-tensor profile swaps from two saved bf16-to-hgn plans.

This reads JSON only. It cannot authenticate source weights, run a conversion,
or produce a recipe accepted by the worker.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re


SCHEMA = "brewery.profile_preview.v1"
MAX_RECIPE_BYTES = 256 * 1024
MAX_PLAN_BYTES = 8 * 1024 * 1024
SPEC = ("1.1.0", "79e75fd4ae7642478823525b0a3212951f584af5")
PROFILES = ("v2", "ht43")
SOURCE_FIELDS = (
    "source_format", "source_shards", "source_tensors", "source_dtypes",
    "canonical_source_payload_bytes",
)
ALTERNATIVES = {
    "hyper_connection": {("Q6G64", "Q8G64")},
    "routed_expert": {("I4r", "Ht"), ("I4r", "Ht3")},
}
LIMITS = [
    "Plans are metadata only. Matching source descriptors do not establish identical "
    "weights or assets. No source or output payloads were read.",
    "Only whole weights with a different encoding in the other built-in profile can "
    "be selected. Required rotation companions follow the weight. No per-expert slices, "
    "arbitrary codecs, or new HGN entries are supported.",
    "HGN size includes payloads, rotation companions, file headers, tensor directories, "
    "and 64-byte alignment. It excludes copied assets, disk reserve, and conversion scratch.",
    "This preview is not a runnable conversion recipe, runtime compatibility test, "
    "quality evaluation, or release artifact.",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_json(path, limit):
    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        require(path.is_file(), "expected a regular JSON file")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit, f"JSON exceeds {limit} bytes")
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
    return sha256(data.encode("ascii")).hexdigest()


def aligned(size, alignment=64):
    return (size + alignment - 1) // alignment * alignment


def integer(value, minimum=1):
    return type(value) is int and minimum <= value < 2**64


def text(value):
    return isinstance(value, str) and bool(value) and value.isascii() and value.isprintable()


def hgn_size(encoding, shape):
    """Cross-check the pinned layout dialect; preview uses the native entries' sizes."""
    n, k = prod(shape[:-1]), shape[-1]
    values = n * k
    if encoding in ("Bf16", "F16", "F32", "I64"):
        return values * {"Bf16": 2, "F16": 2, "F32": 4, "I64": 8}[encoding]
    if encoding == "Q4Cp" and k % 32 == 0:
        return 64 + aligned(values // 2) + n * aligned(k // 16, 16)
    if encoding == "Q6G64" and k % 64 == 0:
        return n * aligned(k // 16 * 13, 16)
    if encoding == "Q8G64" and k % 64 == 0:
        return n * (k + k // 16)
    if encoding == "Fp8G128" and len(shape) == 3 and k == 160:
        return aligned(values) + 4
    if encoding == "I4r" and len(shape) == 3 and shape[1] % 128 == k % 128 == 0:
        return values // 2 + values // 64
    if encoding in ("Ht", "Ht3"):
        if (len(shape) == 2 and all(128 <= d <= 2**24 and d % 128 == 0 for d in shape)
                or shape in ([512, 1280, 2560], [512, 2560, 640])):
            return values * (4 if encoding == "Ht" else 3) // 8
    raise ValueError(f"unsupported HGN encoding or shape: {encoding!r} {shape}")


def source_origin(origin):
    require(isinstance(origin, dict) and len(origin) == 1, "invalid tensor origin")
    kind, fields = next(iter(origin.items()))
    keys = {"Tensor": {"name"}, "Signs": {"key"}, "HtScales": {"tensor", "signs_key"},
            "Ngram": {"shards"}}
    require(isinstance(fields, dict) and keys.get(kind) == fields.keys(),
            f"invalid {kind} origin")
    if kind == "Ngram":
        shards = fields["shards"]
        require(isinstance(shards, list) and shards and all(map(text, shards))
                and len(set(shards)) == len(shards), "invalid n-gram shard names")
    else:
        require(all(map(text, fields.values())), f"invalid {kind} origin names")
    return kind


@dataclass(frozen=True)
class Tensor:
    encoding: str
    shape: list
    payload_bytes: int
    origin: dict
    rotation: dict | None

    def description(self):
        return {"encoding": self.encoding, "shape": self.shape, "payload_bytes": self.payload_bytes}


@dataclass(frozen=True)
class Container:
    identity: str
    data_offset: int
    file_size: int
    tensors: dict


def load_plan(report, profile):
    summary, options, plan = (report[key] for key in ("summary", "options", "plan"))
    require(summary["profile"] == options["profile"] == profile,
            f"expected native {profile} plan")
    require(summary["tensor_payloads_read"] is False
            and (summary["spec_version"], summary["spec_commit"]) == SPEC
            and summary["architecture"] == plan["architecture"] == "qwen4_exp"
            and summary["policy"] == f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1",
            "unsupported native plan version, policy, or architecture")
    assets = plan["assets"]
    require(isinstance(assets, list) and all(map(text, assets))
            and summary["assets"] == assets, "inconsistent asset lists")
    metadata = {field: summary[field] for field in SOURCE_FIELDS}
    dtypes = metadata["source_dtypes"]
    require(metadata["source_format"] in ("hf", "hgn", "gguf")
            and all(integer(metadata[k]) for k in SOURCE_FIELDS
                    if k not in ("source_format", "source_dtypes"))
            and isinstance(dtypes, dict) and all(map(text, dtypes))
            and all(map(integer, dtypes.values()))
            and sum(dtypes.values()) == metadata["source_tensors"], "invalid source metadata")

    containers = {}
    for file, layout, output in zip(plan["files"], report["layouts"], summary["outputs"],
                                    strict=True):
        filename, identity, definitions = (file[key] for key in ("filename", "identity", "tensors"))
        require(text(filename) and Path(filename).name == filename and filename.endswith(".hgn")
                and filename not in containers, f"invalid or duplicate HGN filename: {filename!r}")
        require(text(identity) and len(identity) <= 63
                and layout["identity"] == identity, f"{filename}: inconsistent identity")
        require(isinstance(definitions, list) and 0 < len(definitions) <= 2**20,
                f"{filename}: invalid tensor count")
        data_offset = aligned(104 + 160 * len(definitions))
        require(layout["data_offset"] == data_offset, f"{filename}: invalid data offset")
        tensors, offset, end = {}, data_offset, data_offset
        for definition, entry in zip(definitions, layout["entries"], strict=True):
            name, encoding, shape, origin, rotation = (definition[key] for key in (
                "name", "encoding", "shape", "origin", "rotation"))
            require(text(name) and len(name) <= 95 and name not in tensors,
                    f"{filename}: invalid or duplicate tensor name: {name!r}")
            require(isinstance(shape, list) and 1 <= len(shape) <= 4
                    and all(integer(d) and d < 2**63 for d in shape),
                    f"{name}: invalid shape")
            size = entry["size"]
            require(entry["name"] == name and entry["encoding"] == encoding
                    and entry["shape"] == shape and entry["offset"] == offset
                    and entry["checksum"] == 0 and integer(size) and size == hgn_size(encoding, shape),
                    f"{name}: inconsistent native layout entry")
            kind = source_origin(origin)
            if rotation is not None:
                require(kind == "Tensor" and isinstance(rotation, dict)
                        and rotation.keys() == {"su", "sv"}
                        and all(map(text, rotation.values())) and rotation["su"] != rotation["sv"],
                        f"{name}: invalid rotation references")
            tensors[name] = Tensor(encoding, shape, size, origin, rotation)
            end = offset + size
            offset = aligned(end)
        require(integer(end) and end == layout["file_size"], f"{filename}: inconsistent file size")
        counts = Counter(t.encoding for t in tensors.values())
        require(output == {"filename": filename, "identity": identity, "tensors": len(tensors),
                           "bytes": end, "stores": counts}, f"{filename}: inconsistent plan summary")
        containers[filename] = Container(identity, data_offset, end, tensors)
    require("model.hgn" in containers
            and containers["model.hgn"].identity == f"qwen3.8-flash-next-{profile}"
            and summary["output_hgn_bytes"] == sum(c.file_size for c in containers.values()),
            "inconsistent main profile identity or total HGN size")
    return containers, metadata, assets, {k: v for k, v in options.items() if k != "profile"}


def role_of(filename, name):
    if filename != "model.hgn":
        return None, None
    match = re.fullmatch(r"layers\.(0|[1-9][0-9]*)\.(.+)", name)
    layer, suffix = (int(match[1]), match[2]) if match else (None, name)
    if re.fullmatch(r"(?:attn_hyper_connection|mlp_hyper_connection|hyper_connection_mixer)"
                    r"\.(?:block_inject_weight|input_mix_weight_down|input_mix_weight_up)\.weight",
                    suffix):
        return layer, "hyper_connection"
    if layer is not None and re.fullmatch(r"mlp\.experts\.(?:down_proj|gate_up_proj)\.weight", suffix):
        return layer, "routed_expert"
    return layer, None


def expert_companions(weight, name, tensors):
    """Check the native expert rotation contract before borrowing companion entries."""
    experts, rows, cols = weight.shape
    ht = weight.encoding in ("Ht", "Ht3")
    require(weight.rotation is not None, f"{name}: missing expert rotation")
    su, sv = weight.rotation["su"], weight.rotation["sv"]
    su_shape = [experts, cols] if ht and name.endswith(".down_proj.weight") else [cols]
    sv_shape = [experts, rows] if ht else [rows]
    sv_origin = ({"HtScales": {"tensor": weight.origin["Tensor"]["name"], "signs_key": sv}}
                 if ht else {"Signs": {"key": sv}})
    for key, shape, origin in ((su, su_shape, {"Signs": {"key": su}}), (sv, sv_shape, sv_origin)):
        ref = tensors.get(key)
        require(ref is not None and ref.encoding == "F16" and ref.shape == shape
                and ref.origin == origin and ref.rotation is None,
                f"{name}: incompatible rotation companion {key}")
    return su, sv


def compatible(v2, ht43):
    """Return eligible weights and their changed companions; reject unrelated differences."""
    require(list(v2) == list(ht43), "incompatible container inventories")
    eligible, companions = {}, {}
    for filename, old in v2.items():
        new = ht43[filename]
        a, b = old.tensors, new.tensors
        require(list(a) == list(b) and (filename == "model.hgn" or old.identity == new.identity),
                f"{filename}: incompatible tensor inventory, order, or sidecar identity")
        differences = {name for name in a if a[name] != b[name]}
        refs = Counter(ref for t in a.values() for ref in (t.rotation or {}).values())
        matched = set()
        for name in a:
            before, after = a[name], b[name]
            require(before.rotation == after.rotation, f"{name}: incompatible rotation references")
            if "Tensor" not in before.origin and "Tensor" not in after.origin:
                continue
            require(before.origin == after.origin and before.shape == after.shape,
                    f"{name}: incompatible source tensor or shape")
            if name not in differences:
                continue
            layer, role = role_of(filename, name)
            require((before.encoding, after.encoding) in ALTERNATIVES.get(role, ()),
                    f"{name}: unsupported built-in encoding pair")
            if role == "routed_expert":
                expert_companions(after, name, b)
                for ref in expert_companions(before, name, a):
                    if ref in differences:
                        require(refs[ref] == 1, f"{ref}: shared changed rotation companion")
                        companions.setdefault((filename, name), []).append(ref)
                        matched.add(ref)
            else:
                require(before.rotation is None, f"{name}: unexpected rotation")
            eligible[filename, name] = (layer, role)
            matched.add(name)
        unexplained = differences - matched
        require(not unexplained, f"{filename}: incompatible plan entry: {min(unexplained, default='')}")
    return eligible, companions


def select_overrides(recipe, eligible):
    require(isinstance(recipe, dict)
            and recipe.keys() == {"schema_version", "base_profile", "overrides"}
            and type(recipe["schema_version"]) is int and recipe["schema_version"] == 1,
            "expected recipe schema_version 1 with base_profile and overrides")
    require(recipe["base_profile"] in PROFILES, "base_profile must be v2 or ht43")
    require(isinstance(recipe["overrides"], list), "overrides must be an array")
    selected = set()
    for index, rule in enumerate(recipe["overrides"], start=1):
        prefix = f"override {index}"
        require(isinstance(rule, dict) and rule.keys() == {"select"}
                and isinstance(rule["select"], dict), f"{prefix}: expected only a select object")
        selector = rule["select"]
        matches = set()
        if selector.keys() == {"tensors"}:
            names = selector["tensors"]
            require(isinstance(names, list) and names and all(map(text, names))
                    and len(names) == len(set(names)), f"{prefix}: invalid exact tensor names")
            by_name = {key[1]: key for key in eligible}
            for name in names:
                require(name in by_name, f"{prefix}: no compatible alternative for tensor {name!r}")
                matches.add(by_name[name])
        else:
            require(selector and selector.keys() <= {"layers", "role"},
                    f"{prefix}: expected tensors, layers, or role selector")
            role = selector.get("role")
            require("role" not in selector or role in tuple(ALTERNATIVES),
                    f"{prefix}: supported roles are hyper_connection and routed_expert")
            candidates = {key: layer for key, (layer, kind) in eligible.items()
                          if role is None or role == kind}
            if "layers" in selector:
                layers = selector["layers"]
                require(isinstance(layers, list) and layers and all(integer(n, 0) for n in layers)
                        and len(set(layers)) == len(layers), f"{prefix}: invalid layer indices")
                for layer in layers:
                    group = {key for key, candidate in candidates.items() if candidate == layer}
                    require(group, f"{prefix}: no compatible alternative in layer {layer}")
                    matches.update(group)
            else:
                matches = set(candidates)
        require(matches, f"{prefix}: unmatched selector")
        overlap = selected & matches
        require(not overlap, f"{prefix}: conflicting selectors for {min(overlap, default=('', ''))[1]}")
        selected.update(matches)
    return selected


def preview(recipe, reports):
    v2, source, assets, options = load_plan(reports["v2"], "v2")
    ht43, other_source, other_assets, other_options = load_plan(reports["ht43"], "ht43")
    require(source == other_source and assets == other_assets, "incompatible source metadata")
    require(options == other_options, "incompatible native plan options (other than profile)")
    eligible, companions = compatible(v2, ht43)
    selected = select_overrides(recipe, eligible)
    profile = recipe["base_profile"]
    alternate = "ht43" if profile == "v2" else "v2"
    base, other = (v2, ht43) if profile == "v2" else (ht43, v2)
    entries = {}
    for key in selected:
        entries[key] = ("override", *eligible[key])
        for name in companions.get(key, ()):
            entries[key[0], name] = ("companion", *eligible[key])
    changes = []
    for (filename, name), (kind, layer, role) in sorted(entries.items()):
        changes.append({
            "filename": filename, "tensor": name, "kind": kind, "layer": layer, "role": role,
            "before": base[filename].tensors[name].description(),
            "after": other[filename].tensors[name].description(),
        })
    resolved = {
        "schema_version": 1,
        "base": {"profile": profile, "identity": base["model.hgn"].identity,
                 "plan_sha256": digest(reports[profile])},
        "alternate": {"profile": alternate, "identity": other["model.hgn"].identity,
                      "plan_sha256": digest(reports[alternate])},
        "changes": changes,
    }
    files = []
    for filename, container in base.items():
        sizes = [other[filename].tensors[name].payload_bytes if (filename, name) in entries
                 else t.payload_bytes for name, t in container.tensors.items()]
        predicted = container.data_offset + sum(map(aligned, sizes[:-1])) + sizes[-1]
        require(integer(predicted), f"{filename}: predicted layout exceeds HGN size limit")
        files.append({
            "filename": filename, "base_hgn_bytes": container.file_size,
            "predicted_hgn_bytes": predicted, "delta_hgn_bytes": predicted - container.file_size,
            "base_overhead_bytes": container.file_size
            - sum(t.payload_bytes for t in container.tensors.values()),
            "predicted_overhead_bytes": predicted - sum(sizes),
        })
    totals = {k: sum(f[k] for f in files)
              for k in ("base_hgn_bytes", "predicted_hgn_bytes", "delta_hgn_bytes")}
    return {
        "schema": SCHEMA, "executable": False, "release_eligible": False,
        "source_compatibility": {"status": "metadata_only", "source_identity": "unverified",
                                 "weight_identity": "unverified", "matching_metadata": source},
        "resolved_recipe": resolved, "resolved_recipe_sha256": digest(resolved),
        "size": {**totals, "files": files}, "limits": LIMITS,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Recipe example: {"schema_version":1,"base_profile":"v2",'
               '"overrides":[{"select":{"layers":[0],"role":"hyper_connection"}}]}. '
               'Use role alone or layers alone for grouped selection; use '
               '{"select":{"tensors":["exact.tensor.weight"]}} for named weights.',
    )
    parser.add_argument("--recipe", type=Path, required=True, help="version 1 recipe JSON (256 KiB)")
    parser.add_argument("--v2-plan", type=Path, required=True, help="saved native v2 plan JSON (8 MiB)")
    parser.add_argument("--ht43-plan", type=Path, required=True, help="saved native ht43 plan JSON (8 MiB)")
    args = parser.parse_args()
    try:
        report = preview(read_json(args.recipe, MAX_RECIPE_BYTES),
                         {"v2": read_json(args.v2_plan, MAX_PLAN_BYTES),
                          "ht43": read_json(args.ht43_plan, MAX_PLAN_BYTES)})
    except (KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
