"""Check a real CPU direction job against a generated tiny HGN checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Replace, Sequence
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hgn_trace_fixture import fixture

CONTRAST = "c844b9b32e5547473b0b30268d99dde456657a5c3d5f98d4e92bbeffd4ac7987"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast_fixture(root, cache):
    # The synthetic tokenizer maps each exact chat prompt to four known tokens.
    # The last three include a class token and two shared template tokens.
    prompts = cache / (CONTRAST + ".json")
    assert digest(prompts) == CONTRAST
    vocab = ("<unk>", "poslow", "poshigh", "neglow", "neghigh", "tailY", "tailX", "ignored")
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(vocab)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    counts = {"train": 0, "held_out": 0}
    replacements = []
    for pair in json.loads(prompts.read_bytes())["rows"]:
        split = pair["split"]
        size = ("low", "high")[counts[split] % 2]
        counts[split] += 1
        for label, cls in (("positive", "pos"), ("negative", "neg")):
            turns = pair[label]
            prompt = (f"<|im_start|>system\n{turns['system']}<|im_end|>\n"
                      f"<|im_start|>user\n{turns['user']}<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            replacements.append(Replace(prompt, f"ignored {cls}{size} tailY tailX"))
    assert counts == {"train": 12, "held_out": 8}
    tokenizer.normalizer = Sequence(replacements)
    fixture(root, tokenizer=tokenizer, embedding_rows=[
        (0, -1), (2, 1), (4, 1), (-2, 1), (-4, 1), (0, 1), (1, 0), (1, -1),
    ])


def check_oracle(row):
    # Fixture layers write zero. Each residual stream equals the embedding.
    # With K=3 the positive/negative classes differ only at the first selected
    # token. The stream sum is 4*[+/-x,1], followed by 4*[0,1] and 4*[1,0].
    xs = (2, 4)
    cos = [x / math.hypot(x, 1) for x in xs]
    sin = [1 / math.hypot(x, 1) for x in xs]
    pos = [(1 + c) / 3 for c in cos] * 4
    neg = [(1 - c) / 3 for c in cos] * 4
    margin = mean(pos) - mean(neg)
    cohens_d = margin / math.hypot(stdev(pos), stdev(neg)) * math.sqrt(2)
    expected_negative = [(1 - mean(cos)) / 3, (1 + mean(sin)) / 3] + [0] * 126
    unit_error = max(abs(a - b) for a, b in zip(row["unit_vector"], [1] + [0] * 127, strict=True))
    negative_error = max(abs(a - b) for a, b in zip(row["negative_mean"], expected_negative, strict=True))
    assert unit_error < 1e-7 and negative_error < 1e-7
    for key, expected in {
        "norm": 2 * mean(cos) / 3,
        "held_out_margin": margin,
        "held_out_cohens_d": cohens_d,
        "mean_residual_norm": 4 * (mean(math.hypot(x, 1) for x in xs) + 2) / 3,
    }.items():
        assert math.isclose(row[key], expected, rel_tol=1e-6), (key, row[key], expected)
    return {"expected_cohens_d": cohens_d, "max_unit_error": unit_error,
            "max_negative_mean_error": negative_error}


def exercise(engine, cache, receipt):
    with TemporaryDirectory(prefix="hgn-direction-") as temporary:
        root = Path(temporary)
        contrast_fixture(root, cache)
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
        assert fit["selected_layer"] == 0, fit["selected_layer"]
        for row in fit["layers"]:
            assert row["held_out_cohens_d"] > 1
            assert math.isclose(math.hypot(*row["unit_vector"]), 1, abs_tol=1e-7)
            oracle = check_oracle(row)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"runtime_sha256": result["runtime_sha256"], "model_sha256": sha,
                          "train_pairs": 12, "held_out_pairs": 8, "layers": [0, 1], "last_k": 3,
                          "schema_version": result["schema_version"], "selected_layer": fit["selected_layer"],
                          "held_out_cohens_d": fit["layers"][0]["held_out_cohens_d"],
                          "unit_norm": math.hypot(*fit["layers"][0]["unit_vector"]), "oracle": oracle,
                          "width": 128, "mean_residual_norm": fit["layers"][0]["mean_residual_norm"],
                          "release_qualified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    exercise(args.engine.resolve(), args.cache.resolve(), args.receipt.resolve())
