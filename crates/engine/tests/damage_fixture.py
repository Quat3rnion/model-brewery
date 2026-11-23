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
else:
    assert command == "grad-select" and spec["objective"] == "target_logit"
    data = struct.pack("<32f", *[0.25] * 32)
    (out / "gradient.f32").write_bytes(data)
    (out / "gradient.json").write_text(json.dumps({
        **spec, "schema_version": 1, "shape": [32, 1], "dtype": "f32_le",
        "logit": logit, "backend": "cpu", "placement": {"strict": True, "device": "CPU"},
        "sha256": hashlib.sha256(data).hexdigest(),
    }))
