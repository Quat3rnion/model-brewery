"""CPU regression for the offline trainer command with a real tiny Qwen decoder."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from qsa_train import (
    load_teacher_tokenizer, plan_teacher_devices, teacher_source, teacher_weight_view, tokenizer_source,
)


class TeacherPlacementTest(unittest.TestCase):
    def test_asymmetric_caps_keep_teacher_on_gpus_without_offload(self):
        class MetaModel(torch.nn.Sequential):
            _no_split_modules = ["Linear"]
            all_tied_weights_keys = {}

        # 62 indivisible 1-GiB layers, but the fixture itself allocates no tensors.
        model = MetaModel(*(torch.nn.Linear(32, 2**24, bias=False, device="meta", dtype=torch.bfloat16)
                            for _ in range(62)))
        gib = 1 << 30
        plan = plan_teacher_devices(model, {0: 44 * gib, 1: 22 * gib})
        self.assertEqual(set(plan.values()), {0, 1})
        self.assertEqual(len(plan), 62)
        self.assertGreater(sum(device == 0 for device in plan.values()), 32)
        for device, cap in ((0, 44), (1, 22)):
            self.assertLessEqual(sum(value == device for value in plan.values()), cap)

        # Equal aggregate capacity is no guarantee: the planner reserves room
        # for a whole layer on device zero and must not fall back to disk.
        with self.assertRaisesRegex(RuntimeError, "CPU and disk offload"):
            plan_teacher_devices(model, {0: 40 * gib, 1: 22 * gib})


class TeacherSourcePolicyTest(unittest.TestCase):
    def test_standard_dense_qwen_source(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            config = {"model_type": "qwen3_5", "text_config": {}, "vision_config": {}}
            payload = json.dumps(config).encode("utf-8")
            (root / "config.json").write_bytes(payload)
            save_file({"example": torch.tensor([1, 2, 3])}, root / "model.safetensors")
            validated = teacher_source(root)
            self.assertEqual(validated.config, config)
            self.assertEqual(validated.config_sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(validated.shards, ("model.safetensors",))
            with teacher_weight_view(root, validated, root) as view:
                self.assertEqual({p.name for p in view.iterdir()}, {"model.safetensors"})
                with safe_open(view / "model.safetensors", framework="pt") as file:
                    tensor = file.get_slice("example")[:]
            self.assertFalse(view.exists())
            self.assertEqual(tensor.tolist(), [1, 2, 3])

    def test_rejects_alternate_checkpoint_or_config_selectors(self):
        config = {"model_type": "qwen3_5", "text_config": {}, "vision_config": {}}
        for changes, marker, reason in (
            ({"configuration_files": ["config.5.22.0.json"]}, None, "configuration_files"),
            ({"transformers_weights": "alternate.safetensors"}, None, "transformers_weights"),
            ({}, "adapter_config.json", "adapter"),
            ({}, "shard-00001-model-00001-of-00001.safetensors", "distributed checkpoint"),
        ):
            with self.subTest(changes=changes, marker=marker), tempfile.TemporaryDirectory(dir="/arboretum") as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(config | changes), encoding="utf-8")
                (root / "model.safetensors").touch()
                if marker:
                    (root / marker).touch()
                with self.assertRaisesRegex(ValueError, reason):
                    teacher_source(root)

    def test_disallows_versioned_tokenizer_selector(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text(
                json.dumps({"fast_tokenizer_files": ["tokenizer.5.22.0.json"]}), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fast_tokenizer_files"):
                tokenizer_source(root)


class OfflineQsaTrainingTest(unittest.TestCase):
    def test_trains_and_exports_only_indexers_from_frozen_qwen(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            teacher = root / "teacher"
            tokenizer = Tokenizer(WordLevel({
                "[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5,
            }, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            direct_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]",
            )
            direct_tokenizer.save_pretrained(teacher)
            config = Qwen3_5Config(
                text_config={
                    "vocab_size": 16, "hidden_size": 32, "intermediate_size": 48,
                    "num_hidden_layers": 3, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 128,
                    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                        "partial_rotary_factor": 0.5, "mrope_section": [12, 10, 10]},
                    "layer_types": ["linear_attention", "full_attention", "full_attention"],
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                },
                vision_config={
                    "hidden_size": 16, "intermediate_size": 24, "num_heads": 2,
                    "depth": 1, "out_hidden_size": 32, "num_position_embeddings": 16,
                },
            )
            torch.manual_seed(42)
            Qwen3_5ForConditionalGeneration(config).save_pretrained(teacher, max_shard_size="64KB")
            loaded_tokenizer = load_teacher_tokenizer(teacher, config, root)
            self.assertEqual(loaded_tokenizer.encode("a d c", add_special_tokens=False),
                             direct_tokenizer.encode("a d c", add_special_tokens=False))
            self.assertEqual(loaded_tokenizer.eos_token_id, direct_tokenizer.eos_token_id)
            self.assertEqual(list(root.iterdir()), [teacher])
            digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in teacher.iterdir()}
            train, heldout, output = root / "train.txt", root / "heldout.txt", root / "output"
            train.write_text("a a a a b b b b c c c c d d d d\n", encoding="utf-8")
            heldout.write_text("c a a c d d a d a d a d c c d c\n", encoding="utf-8")
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
            args = [
                sys.executable, str(Path(__file__).with_name("qsa_train.py")),
                "--source", str(teacher), "--train-data", str(train), "--eval-data", str(heldout),
                "--output-dir", str(output), "--steps", "3", "--seq-len", "16",
                "--seed", "18", "--lr", "0.002", "--query-chunk", "3", "--budget", "4",
                "--device", "cpu",
            ]
            result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in teacher.iterdir()}, digests)
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"metrics.json", "qsa-indexers.safetensors"})
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["status"], "passed")
            self.assertEqual(metrics["steps"], 3)
            self.assertEqual(metrics["full_attention_layers"], [1, 2])
            self.assertEqual(metrics["source_config_sha256"], digests["config.json"])
            self.assertEqual(metrics["before"]["queries"], 26)
            self.assertEqual(metrics["after"]["capture_queries"], 18)
            self.assertNotEqual(metrics["before"]["kl"], metrics["after"]["kl"])
            self.assertGreater(metrics["indexer_update_l2"], 0)
            for phase in ("before", "after"):
                self.assertTrue(0 < metrics[phase]["capture"] < 1)
            with safe_open(output / "qsa-indexers.safetensors", framework="pt") as bundle:
                self.assertEqual(bundle.metadata()["source_config_sha256"], digests["config.json"])
                shapes = {"q_proj.weight": (512, 32), "k_proj.weight": (128, 32),
                          "q_norm": (128,), "k_norm": (128,), "k_block_norm": (128,)}
                expected = {f"model.language_model.layers.{i}.self_attn._qsa_indexer.{key}": shape
                            for i in (1, 2) for key, shape in shapes.items()}
                self.assertEqual(set(bundle.keys()), set(expected))
                for key, shape in expected.items():
                    tensor = bundle.get_tensor(key)
                    self.assertEqual(tensor.shape, shape)
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(tensor).all())

            # With a budget covering all blocks, there is no retrieval evidence.
            # Change the chunk size too: KL must still match on the same teacher.
            uncompetitive = root / "uncompetitive"
            args[args.index("--output-dir") + 1] = str(uncompetitive)
            args[args.index("--budget") + 1] = "2048"
            args[args.index("--query-chunk") + 1] = "16"
            result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            other = json.loads((uncompetitive / "metrics.json").read_text())
            for phase in ("before", "after"):
                self.assertIsNone(other[phase]["capture"])
                self.assertEqual(other[phase]["capture_queries"], 0)
                self.assertIn("capture_unavailable_reason", other[phase])
                self.assertEqual(other[phase]["queries"], metrics[phase]["queries"])
                self.assertAlmostEqual(other[phase]["kl"], metrics[phase]["kl"], places=5)

            train_line = train.read_text(encoding="utf-8").rstrip("\n")
            eval_line = heldout.read_text(encoding="utf-8").rstrip("\n")
            for name, train_bytes, eval_bytes in (
                ("reserialized", f"{train_line}\n", f"\r\n{train_line}\r\n \r\n"),
                # Share only the second of two full batches.
                ("shared-full-batch", f"{train_line} {eval_line}\n",
                 f"d {train_line.split(' ', 1)[1]} {eval_line}\n"),
            ):
                with self.subTest(name=name):
                    rejected = root / name
                    train.write_bytes(train_bytes.encode("utf-8"))
                    heldout.write_bytes(eval_bytes.encode("utf-8"))
                    args[args.index("--output-dir") + 1] = str(rejected)
                    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=40)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("share a packed token sequence", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertFalse((rejected / "metrics.json").exists())
                    self.assertFalse((rejected / "qsa-indexers.safetensors").exists())

            train.write_text(f"{train_line}\n", encoding="utf-8")
            heldout.write_text(f"{eval_line}\n", encoding="utf-8")
            shard_index = teacher / "model.safetensors.index.json"
            manifest = json.loads(shard_index.read_text())
            first_tensor = next(iter(manifest["weight_map"]))
            manifest["weight_map"][first_tensor] = "missing.safetensors"
            shard_index.write_text(json.dumps(manifest))
            rejected = root / "missing-shard"
            args[args.index("--output-dir") + 1] = str(rejected)
            result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=40)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Teacher SafeTensors shard is not a regular local member", result.stderr)
            self.assertFalse((rejected / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
