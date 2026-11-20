"""Bounded file and routed-MLP parity tests for offline expert surgery."""

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools" / "expert_safetensors.py"
PLANNER = Path(__file__).resolve().with_name("expert_plan.py")
TRUNK = "model.language_model.layers.0.mlp."
MTP = "mtp.layers.0.mlp."
INDEX = "model.safetensors.index.json"
WEIGHTS = "model.safetensors"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_weights(path, entries):
    header, payload = {"__metadata__": {"format": "pt"}}, bytearray()
    for name, (shape, data) in entries.items():
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data) * 4]}
        payload.extend(struct.pack(f"<{len(data)}f", *data))
    descriptor = json.dumps(header).encode()
    descriptor += b" " * (-len(descriptor) % 8)
    path.write_bytes(struct.pack("<Q", len(descriptor)) + descriptor + payload)


def read_weights(path):
    raw = path.read_bytes()
    length, = struct.unpack_from("<Q", raw)
    header = json.loads(raw[8:8 + length])
    result = {}
    assert header.pop("__metadata__") == {"format": "pt"}
    for key, spec in header.items():
        lo, hi = spec["data_offsets"]
        assert spec["dtype"] == "F32"
        result[key] = (spec["shape"], struct.unpack(
            f"<{(hi - lo) // 4}f", raw[8 + length + lo:8 + length + hi]))
    return result


def entries(prefix):
    return {prefix + key: value for key, value in {
        "experts.gate_up_proj": ([3, 2, 2], [1, 2, 3, 4, 2, 4, 6, 8, 7, -3, 8, 5]),
        "experts.down_proj": ([3, 2, 1], [2, 1, 4, 3, 6, 5]),
        "experts.down_proj_scale": ([3, 2, 1], [1, 2, 3, 4, 5, 6]),
        "gate.weight": ([3, 2], [1, 3, 0, 4, 2, 0]),
        "gate.bias": ([3], [-5, 0, 1]),
        "shared_expert.up_proj.weight": ([1, 2], [42, -7]),
    }.items()}


def config():
    return {"model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {"model_type": "qwen4_exp_text", "num_hidden_layers": 1,
                            "num_experts": 3, "num_experts_per_tok": 2, "hidden_size": 2,
                            "moe_intermediate_size": 1, "mtp_num_hidden_layers": 1,
                            "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}}}


def layer_forward(weights, prefix, token):
    def tensor(name):
        return weights[prefix + name][1]

    def dot(values):
        return sum(x * y for x, y in zip(values, token, strict=True))

    bias = tensor("gate.bias")
    gate = tensor("gate.weight")
    logits = [dot(gate[2*i:2*i+2]) + b for i, b in enumerate(bias)]
    selected = sorted(range(len(bias)), key=logits.__getitem__, reverse=True)[:2]
    scores = [math.exp(logits[i] - logits[selected[0]]) for i in selected]
    result = [dot(tensor("shared_expert.up_proj.weight"))] * 2
    gate_up, down, scales = (tensor("experts." + name) for name in
                             ("gate_up_proj", "down_proj", "down_proj_scale"))
    for i, score in zip(selected, scores, strict=True):
        g = dot(gate_up[4*i:4*i+2])
        value = g / (1 + math.exp(-g)) * dot(gate_up[4*i+2:4*i+4])
        for j in range(2):
            result[j] += score / sum(scores) * value * down[2*i+j] * scales[2*i+j]
    return selected, result


class ExpertSafeTensorsTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="expert-safetensors-test.", dir="/arboretum")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)

    def cli(self, *args, ok=True):
        run = subprocess.run([sys.executable, str(CLI), *map(str, args)],
                             text=True, capture_output=True, check=False)
        if ok:
            self.assertEqual(run.returncode, 0, run.stderr)
        else:
            self.assertNotEqual(run.returncode, 0, run.stdout)
        return run

    def standalone(self):
        source, cfg = self.root / "original.safetensors", self.root / "source-config.json"
        write_weights(source, entries(TRUNK) | entries(MTP))
        cfg.write_text(json.dumps(config()))
        pin = ("--config", cfg, "--config-sha256", sha(cfg.read_bytes()))
        return source, pin

    def plan(self, source_digest, maps):
        path = self.root / "plan.json"
        path.write_text(json.dumps({"schema": "brewery.expert-map.v1",
                                    "source_sha256": source_digest, "layers": maps,
                                    "provenance": {"profile": "fixture"}}))
        return ("--plan", path, "--plan-sha256", sha(path.read_bytes()))

    def indexed(self):
        source = self.root / "indexed"
        source.mkdir()
        (source / "assets").mkdir()
        (source / "assets" / "tokenizer.json").write_bytes(b"opaque sidecar\n")
        cfg = config()
        cfg["text_config"]["mtp"]["num_experts"] = 3
        (source / "config.json").write_text(json.dumps(cfg))
        all_entries = entries(TRUNK) | entries(MTP) | {"lm_head.weight": ([2], [19, 21])}
        shards = [{}, {}]
        for i, (name, value) in enumerate(all_entries.items()):
            shards[i % 2][name] = value
        mapping = {}
        for i, shard in enumerate(shards, 1):
            basename = f"model-0000{i}-of-00002.safetensors"
            write_weights(source / basename, shard)
            mapping.update(dict.fromkeys(shard, basename))
        total = sum(4 * len(value) for _, value in all_entries.values())
        (source / INDEX).write_text(json.dumps({"metadata": {"total_size": total, "owner": "fixture"},
                                                "weight_map": mapping}))
        return source, mapping, total

    def test_standalone_reorder_preserves_routed_forward_and_shared_expert(self):
        source, pin = self.standalone()
        original = read_weights(source)
        digest = sha(source.read_bytes())
        self.assertEqual(self.cli("source-digest", "--source", source, *pin).stdout.strip(), digest)
        dest = self.root / "reordered"
        self.cli("apply", "--source", source, *pin,
                 *self.plan(digest, {"0": [2, 0, 1]}), "--output", dest)
        transformed = read_weights(dest / WEIGHTS)
        self.assertEqual(transformed[TRUNK + "gate.bias"][1], (1, -5, 0))
        for token in ((.25, 1), (2, -1), (-1, .5)):
            old_ids, before = layer_forward(original, TRUNK, token)
            new_ids, after = layer_forward(transformed, TRUNK, token)
            self.assertEqual([[2, 0, 1][i] for i in new_ids], old_ids)
            self.assertEqual(after, before)
        for name in original:
            if name.startswith(MTP) or "shared_expert" in name:
                self.assertEqual(original[name], transformed[name])
        self.assertEqual(json.loads((dest / "config.json").read_text()), config())
        self.assertEqual(sha(source.read_bytes()), digest)

    def test_indexed_prune_compacts_two_shards_updates_config_and_keeps_shared_weights(self):
        source, mapping, total = self.indexed()
        files = {p.relative_to(source).as_posix(): p.read_bytes()
                 for p in source.rglob("*") if p.is_file()}
        digest = sha(b"".join(f"{n}\0{len(files[n])}\0{sha(files[n])}\n".encode()
                              for n in sorted(files)))
        self.assertEqual(self.cli("source-digest", "--source", source).stdout.strip(), digest)
        dest = self.root / "pruned"
        self.cli("apply", "--source", source, *self.plan(digest, {"0": [2, 0], "1": [0, 1]}),
                 "--output", dest)
        cfg = json.loads((dest / "config.json").read_text())["text_config"]
        self.assertEqual(cfg["num_experts"], 2)
        self.assertEqual(cfg["mtp"]["num_experts"], 2)
        self.assertEqual(cfg["num_experts_per_tok"], 2)
        index = json.loads((dest / INDEX).read_text())
        self.assertEqual(index["weight_map"], mapping)
        # Two layers each lose 4+2+2+2+1 float32 values; no source tensor is omitted.
        self.assertEqual(index["metadata"], {"total_size": total - 88, "owner": "fixture"})
        result = {}
        for member in set(mapping.values()):
            self.assertLess((dest / member).stat().st_size, len(files[member]))
            result.update(read_weights(dest / member))
        self.assertEqual(result[TRUNK + "experts.gate_up_proj"], ([2, 2, 2], (7, -3, 8, 5, 1, 2, 3, 4)))
        self.assertEqual(result[TRUNK + "experts.down_proj"], ([2, 2, 1], (6, 5, 2, 1)))
        self.assertEqual(result[TRUNK + "experts.down_proj_scale"], ([2, 2, 1], (5, 6, 1, 2)))
        self.assertEqual(result[MTP + "gate.weight"], ([2, 2], (1, 3, 0, 4)))
        self.assertEqual(result[MTP + "gate.bias"], ([2], (-5, 0)))
        self.assertEqual(result[MTP + "experts.down_proj"], ([2, 2, 1], (2, 1, 4, 3)))
        for prefix in (TRUNK, MTP):
            self.assertEqual(result[prefix + "shared_expert.up_proj.weight"], ([1, 2], (42, -7)))
        self.assertEqual(result["lm_head.weight"], ([2], (19, 21)))
        self.assertEqual((dest / "assets" / "tokenizer.json").read_bytes(), files["assets/tokenizer.json"])
        self.assertEqual({n: (source / n).read_bytes() for n in files}, files)

    def test_reviewed_plan_config_and_source_pins_are_required(self):
        source, pin = self.standalone()
        dest = self.root / "refused"
        original = source.read_bytes()
        args = self.plan(sha(original), {"0": [2, 0, 1]})
        for command, error in (
            (args[:2] + pin, "--plan-sha256"),
            (args, "standalone SafeTensors requires"),
            (args + (*pin[:-1], "0" * 64), "standalone config SHA-256 mismatch"),
            ((*args[:-1], "0" * 64) + pin, "reviewed plan SHA-256 mismatch"),
        ):
            with self.subTest(error=error):
                run = self.cli("apply", "--source", source, *command, "--output", dest, ok=False)
                self.assertIn(error, run.stderr)
                self.assertFalse(dest.exists())
        source.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        run = self.cli("apply", "--source", source, *args, *pin, "--output", dest, ok=False)
        self.assertIn("SafeTensors source SHA-256 mismatch", run.stderr)
        self.assertFalse(dest.exists())

    def test_refuses_ambiguous_or_invalid_maps_and_preexisting_destinations(self):
        source, pin = self.standalone()
        dest = self.root / "refused"
        digest = sha(source.read_bytes())
        for maps, error in (
            ({"0": [0, 0, 1]}, "invalid expert IDs"),
            ({"0": [True, 0, 2]}, "invalid expert IDs"),
            ({"0": [2, 0, 3]}, "invalid expert IDs"),
            ({"0": [1], "1": [1]}, "below top-k"),
            ({"0": [2, 0], "1": [2, 0, 1]}, "counts must agree"),
            ({"0": [2, 0]}, "requires every routed trunk and MTP layer"),
            ({"2": [2, 0, 1]}, "unknown expert layer ID"),
        ):
            with self.subTest(error=error, maps=maps):
                run = self.cli("apply", "--source", source, *self.plan(digest, maps),
                               *pin, "--output", dest, ok=False)
                self.assertIn(error, run.stderr)
                self.assertFalse(dest.exists())
        dest.mkdir()
        marker = dest / "keep"
        marker.write_text("untouched")
        run = self.cli("apply", "--source", source, *self.plan(digest, {"0": [2, 0, 1]}),
                       *pin, "--output", dest, ok=False)
        self.assertIn("destination already exists", run.stderr)
        self.assertEqual(list(dest.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "untouched")
        self.assertEqual(sha(source.read_bytes()), digest)

    def test_rejects_unknown_expert_tensors_even_on_unmapped_mtp(self):
        source, pin = self.standalone()
        dest = self.root / "refused"
        for name, shape, data in (
            (MTP + "experts.down_proj", [2, 3, 1], [0] * 6),
            (MTP + "experts.unclear_scale", [3], [0] * 3),
            (MTP + "experts.gate_up_proj_scale", [2, 3], [0] * 6),
            ("model.layers.0.mlp.gate.weight", [3, 2], [0] * 6),
        ):
            with self.subTest(name=name, shape=shape):
                write_weights(source, entries(TRUNK) | entries(MTP) | {name: (shape, data)})
                run = self.cli("apply", "--source", source,
                               *self.plan(sha(source.read_bytes()), {"0": [2, 0, 1]}),
                               *pin, "--output", dest, ok=False)
                self.assertIn("unsupported", run.stderr)
                self.assertIn(name, run.stderr)
                self.assertFalse(dest.exists())

    def test_rejects_grouped_routing_config(self):
        source, pin = self.standalone()
        cfg_path = pin[1]
        cfg = config()
        cfg["text_config"]["n_group"] = 3
        cfg_path.write_text(json.dumps(cfg))
        dest = self.root / "refused"
        run = self.cli("apply", "--source", source,
                       *self.plan(sha(source.read_bytes()), {"0": [2, 0, 1]}),
                       *pin[:-1], sha(cfg_path.read_bytes()), "--output", dest, ok=False)
        self.assertIn("grouped expert routing", run.stderr)
        self.assertFalse(dest.exists())

    def test_reviewed_top_k_must_match_native_config(self):
        source, pin = self.standalone()
        dest = self.root / "refused"
        for values in ({"top_k": 1}, {"top_k": True}, {"top_k": 2, "review": {"top_k": 1}}):
            with self.subTest(values=values):
                _, path, _, _ = self.plan(sha(source.read_bytes()), {"0": [2, 0, 1]})
                path.write_text(json.dumps(json.loads(path.read_bytes()) | values))
                run = self.cli("apply", "--source", source, "--plan", path,
                               "--plan-sha256", sha(path.read_bytes()), *pin, "--output", dest, ok=False)
                self.assertIn("top_k", run.stderr)
                self.assertFalse(dest.exists())
