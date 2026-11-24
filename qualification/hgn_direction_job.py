"""Check a real CPU direction job against a generated tiny HGN checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hgn_trace_fixture import fixture


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exercise(engine, cache):
    with TemporaryDirectory(prefix="hgn-direction-") as temporary:
        root = Path(temporary)
        fixture(root)
        model = root / "model.hgn"
        sha = digest(model)
        assets = {name: {"path": str(root / leaf), "sha256": digest(root / leaf)}
                  for name, leaf in [("config", "config.json"), ("sidecar", "sidecar.hgn"),
                                     ("tokenizer", "tokenizer.json")]}
        registry = root / "assets.json"
        registry.write_text(json.dumps({sha: assets}))
        python = Path(sys.executable)
        env = {
            "BREWERY_CONTRAST_CACHE": str(cache),
            "BREWERY_HGN_TRACE_PYTHON": str(python),
            "BREWERY_HGN_TRACE_PYTHON_SHA256": digest(python),
            "BREWERY_HGN_TRACE_ASSETS": str(registry),
            "BREWERY_CPU_THREADS": "1",
        }
        request = {
            "kind": "behavior_direction",
            "inputs": [{"path": str(model), "kind": "hgn-v2", "id": "tiny", "host": "local", "sha256": sha}],
            "options": {"behavior": "slop", "model_sha256": sha, "layers": [0, 1], "last_k": 3},
        }
        process = subprocess.run([engine], input=json.dumps(request), cwd=root,
                                 env=env, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr)
        result = json.loads(process.stdout)
        assert result["schema_version"] == 2 and not result["release_qualified"]
        assert result["model_sha256"] == sha and result["runtime"] == "hgn_cpu"
        assert result["last_k"] == 3 and result["position"] is None
        assert result["assets_sha256"] == {name: pin["sha256"] for name, pin in assets.items()}
        fit = result["fit"]
        assert (fit["train_pairs"], fit["held_out_pairs"]) == (12, 8)
        assert [row["layer"] for row in fit["layers"]] == [0, 1]
        for row in fit["layers"]:
            assert row["negative_mean"] == [0.0883883461356163] * 128
            assert abs(row["mean_residual_norm"] - 45.254833995939045) < 1e-11
            assert row["unit_vector"] is None and row["held_out_cohens_d"] is None
        print(json.dumps({"runtime_sha256": result["runtime_sha256"], "model_sha256": sha,
                          "train_pairs": 12, "held_out_pairs": 8, "layers": [0, 1], "last_k": 3,
                          "width": 128, "mean_residual_norm": fit["layers"][0]["mean_residual_norm"],
                          "release_qualified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    exercise(args.engine.resolve(), args.cache.resolve())
