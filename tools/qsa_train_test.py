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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration


class OfflineQsaTrainingTest(unittest.TestCase):
    def test_trains_and_exports_only_indexers_from_frozen_qwen(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            teacher = root / "teacher"
            tokenizer = Tokenizer(WordLevel({
                "[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5,
            }, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]",
            ).save_pretrained(teacher)
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
            Qwen3_5ForConditionalGeneration(config).save_pretrained(teacher)
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


if __name__ == "__main__":
    unittest.main()
