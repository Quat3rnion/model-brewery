#!/usr/bin/python3
import hashlib
import json
import pathlib
import struct
import sys

command, *args = sys.argv[1:]
opts = dict(zip(args[::2], args[1::2]))
if command == "tokenize":
    print('{"tokens":[1,2]}')
    sys.exit(0)
spec = json.loads(pathlib.Path(opts["--spec"]).read_text())
out = pathlib.Path(opts["--out"])
out.mkdir()
model = pathlib.Path(opts["--model"]).read_bytes()
# Real GGUF fixture payload: Q8_0 decodes to 1; BF16 encodes 2.
logit = 8 if model[-34:] == bytes([0, 0x38] + [2] * 32) else 16
if command == "trace":
    (out / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "kind": "trace", "spec": spec, "model": {},
        "tokens": [{"position": 0, "token_id": 1}, {"position": 1, "token_id": 2}],
        "tensors": [], "missing": [], "interventions": [], "logit_lens": [],
        "model_logits": [{"position": 1, "top": [{"token_id": 3, "logit": logit}] +
                         [{"token_id": 100 + i, "logit": -i} for i in range(99)]}],
    }))
elif command == "grad-corpus":
    corpus = pathlib.Path(opts["--corpus"]).read_bytes()
    assert hashlib.sha256(corpus).hexdigest() == spec["corpus_sha256"]
    assert spec["weights"] == ["blk.0.attn_q.weight"]
    assert opts["--backend"] == "cpu"
    assert model.endswith(struct.pack("<32f", *[2.0] * 32))
    data = struct.pack("<32f", *[0.25] * 32)
    mode = "__FIXTURE_MODE__"
    (out / "grad-0.f32").write_bytes(data)
    (out / "gradient.json").write_text(json.dumps({
        "schema_version": 1, "objective": "mean_next_token_nll",
        "corpus_sha256": spec["corpus_sha256"],
        "tokens": [[1] * 15] * 15 + [[1] * 13], "observation_count": 222, "loss": 1.25,
        "backend": "cpu", "placement": {"strict": True, "device": "CPU"}, "dtype": "f32_le",
        "gradients": [{
            "name": spec["weights"][0], "shape": [1, 32] if mode == "bad-shape" else [32, 1],
            "file": "../grad-0.f32" if mode == "bad-path" else "grad-0.f32",
            "sha256": "a" * 64 if mode == "bad-hash" else hashlib.sha256(data).hexdigest(),
        }],
    }))
else:
    assert command == "grad-select" and spec["objective"] == "target_logit"
    data = struct.pack("<32f", *[0.25] * 32)
    (out / "gradient.f32").write_bytes(data)
    (out / "gradient.json").write_text(json.dumps({
        **spec, "schema_version": 1, "shape": [32, 1], "dtype": "f32_le",
        "logit": logit, "backend": "cpu", "placement": {"strict": True, "device": "CPU"},
        "sha256": hashlib.sha256(data).hexdigest(),
    }))
