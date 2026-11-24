"""Run the HGN CLI on a two-layer, locally generated HGN fixture."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).parent / "hgn_trace"


class TraceTest(unittest.TestCase):
    def test_token_probe_does_not_open_model(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            spec = {"layers": [], "roles": [], "positions": [0], "top_k": 0, "byte_cap": 2048}
            (path / "spec.json").write_text(json.dumps(spec))
            subprocess.run([
                sys.executable, str(ROOT / "trace.py"), "--model", str(path / "not-opened.hgn"),
                "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                "--out", str(path / "probe"), "--prompt", "a b", "--threads", "1",
            ], check=True)
            m = json.loads((path / "probe/manifest.json").read_text())
            self.assertEqual([t["token_id"] for t in m["tokens"]], [1, 2])
            self.assertEqual(m["tensors"], [])
            self.assertEqual(m["model_logits"], [])

    def test_spec_selects_full_streams_and_logit_views(self):
        from hgn_trace_fixture import fixture

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            spec = {"layers": [1, 0, 1], "roles": ["l_out", "final_norm"],
                    "positions": [1], "top_k": 2, "byte_cap": 8192}
            args = [sys.executable, str(ROOT / "trace.py"), "--model", str(path / "model.hgn"),
                    "--sidecar", str(path / "sidecar.hgn"), "--config", str(path / "config.json"),
                    "--tokenizer", str(path / "tokenizer.json"), "--spec", str(path / "spec.json"),
                    "--out", str(path / "trace"), "--prompt", "a b", "--threads", "1"]
            (path / "spec.json").write_text(json.dumps(spec))
            subprocess.run(args, check=True)
            out = path / "trace"
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["tokens"], [{"position": 0, "token_id": 1},
                                           {"position": 1, "token_id": 2}])
            self.assertEqual(m["spec"]["layers"], [0, 1])
            self.assertEqual(m["missing"], [{"layer": 0, "role": "final_norm", "position": 1}])
            self.assertCountEqual([(t["layer"], t["role"], t["shape"]) for t in m["tensors"]],
                                  [(0, "l_out", [4, 128]), (1, "l_out", [4, 128]),
                                   (1, "final_norm", [128])])
            for tensor in m["tensors"]:
                raw = (out / tensor["file"]).read_bytes()
                self.assertEqual(len(raw), tensor["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), tensor["sha256"])
            self.assertEqual(m["model_logits"][0]["top"], m["logit_lens"][-1]["top"])
            self.assertEqual([x["token_id"] for x in m["model_logits"][0]["top"]], [0, 1])
            residual = (out / m["tensors"][0]["file"]).read_bytes()
            self.assertEqual(struct.unpack("<512f", residual), (1.0,) * 512)
            spec["byte_cap"] = 2048
            (path / "spec.json").write_text(json.dumps(spec))
            args[args.index("--out") + 1] = str(path / "too-small")
            result = subprocess.run(args, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"byte cap", result.stderr)
            self.assertFalse((path / "too-small" / "manifest.json").exists())
            args[args.index("--prompt") + 1] = " ".join(["a"] * 513)
            result = subprocess.run(args, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"512", result.stderr)
