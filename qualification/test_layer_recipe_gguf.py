"""Qualify the source-bound layer recipe CLI with real SafeTensors inputs."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools/gguf-export-hf"
SCHEMA = "brewery.layer-quant.v1"
VOCAB_SIZE = 257
GPT2_VOCAB_SIZE = 50257
DECODE_SHA256 = "f3aaaef689f3c6f329bdea5e68ca397c2a0f3255ab0397515a35c301799d0423"
GPT2_TOKENIZER_SHA256 = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
MATRICES = {
    "attn.c_attn": ("attn_qkv", (32, 96)),
    "attn.c_proj": ("attn_output", (32, 32)),
    "mlp.c_fc": ("ffn_up", (32, 128)),
    "mlp.c_proj": ("ffn_down", (128, 32)),
}
NORMS = {"ln_1": "attn_norm", "ln_2": "ffn_norm"}


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(directory):
    manifest = "".join(
        f"{entry.name}\0{entry.stat().st_size}\0{sha256(entry)}\n"
        for entry in sorted(directory.iterdir())
    )
    return hashlib.sha256(manifest.encode()).hexdigest()


def safetensors(path, tensors):
    payload, header = bytearray(), {}
    for name, (shape, value, *types) in tensors:
        dtype = types[0] if types else "F32"
        start = len(payload)
        payload += struct.pack({"F16": "<e", "F32": "<f"}[dtype], value) * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def read_gguf(path):
    """Independent GGUF v3 reader for the fixture's four GGML types."""
    with path.open("rb") as f:
        def uint(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(uint("<Q")).decode("utf-8")

        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                item, count = uint("<I"), uint("<Q")
                if item == 8:
                    for _ in range(count):
                        string()
                else:
                    f.seek(widths[item] * count, os.SEEK_CUR)
                return None
            return int.from_bytes(f.read(widths[kind]), "little")

        if f.read(4) != b"GGUF" or uint("<I") != 3:
            raise ValueError("expected a GGUF v3 artifact")
        tensors, metadata = uint("<Q"), uint("<Q")
        props = {string(): value(uint("<I")) for _ in range(metadata)}
        entries = {}
        for _ in range(tensors):
            name = string()
            shape = tuple(uint("<Q") for _ in range(uint("<I")))
            dtype, offset = uint("<I"), uint("<Q")
            if name in entries:
                raise ValueError(f"duplicate GGUF tensor {name}")
            entries[name] = shape, dtype, offset
        alignment = props.get("general.alignment", 32)
        start = (f.tell() + alignment - 1) // alignment * alignment
        result = {}
        for name, (shape, dtype, offset) in entries.items():
            size = math.prod(shape) // 32 * 34 if dtype == 8 else math.prod(shape) * {0: 4, 1: 2, 30: 2}[dtype]
            f.seek(start + offset)
            payload = f.read(size)
            if len(payload) != size:
                raise ValueError(f"truncated GGUF tensor {name}")
            result[name] = shape, dtype, payload
        return result


class GgufLayerRecipe(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/arboretum")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = self.root / "recipe.json"
        self.tensors = {}
        self.vectors = {}
        self.matrices = {}
        for src, target, shape in (
            ("transformer.wte.weight", "token_embd.weight", (VOCAB_SIZE, 32)),
            ("transformer.wpe.weight", "position_embd.weight", (128, 32)),
            ("transformer.ln_f.weight", "output_norm.weight", (32,)),
            ("transformer.ln_f.bias", "output_norm.bias", (32,)),
            ("lm_head.weight", "output.weight", (VOCAB_SIZE, 32)),
        ):
            self.add_tensor(src, target, shape)
        for layer in range(2):
            for src, (target, shape) in MATRICES.items():
                self.add_tensor(f"transformer.h.{layer}.{src}.weight", f"blk.{layer}.{target}.weight", shape)
                self.add_tensor(f"transformer.h.{layer}.{src}.bias", f"blk.{layer}.{target}.bias", shape[-1:])
            for src, target in NORMS.items():
                for part in ("weight", "bias"):
                    self.add_tensor(f"transformer.h.{layer}.{src}.{part}", f"blk.{layer}.{target}.{part}", (32,))

    def add_tensor(self, source, target, shape):
        value = (len(self.tensors) + 1) / 128
        if len(shape) == 2 and target.startswith("blk.1."):
            # Exactly representable F16 values that would change after a
            # premature BF16 roundtrip. Use different values for each matrix.
            value += 1 / 8192
        self.tensors[source] = shape, value
        if len(shape) == 1:
            self.vectors[target] = struct.pack("<f", value) * shape[0]
        else:
            self.matrices[target] = value

    def make_source(self, indexed, tokenizer_assets=None):
        assets = self.root / ("indexed-hf" if indexed else "explicit-assets")
        assets.mkdir()
        vocab_size = GPT2_VOCAB_SIZE if tokenizer_assets else VOCAB_SIZE
        end_token = vocab_size - 1
        write_json(assets / "config.json", dict(
            architectures=["GPT2LMHeadModel"], model_type="gpt2", n_ctx=128, n_positions=128,
            n_embd=32, n_head=4, n_layer=2, layer_norm_epsilon=1e-5, vocab_size=vocab_size,
            activation_function="gelu_new", tie_word_embeddings=False,
            bos_token_id=end_token, eos_token_id=end_token,
        ))
        if tokenizer_assets:
            # Actual export checks a known tokenizer's encoded token sequence;
            # operand data can stay synthetic but tokenizer bytes cannot.
            self.assertEqual(sha256(tokenizer_assets / "tokenizer.json"), GPT2_TOKENIZER_SHA256)
            for asset in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
                shutil.copyfile(tokenizer_assets / asset, assets / asset)
        else:
            write_json(assets / "tokenizer.json", {
                "pre_tokenizer": {"type": "ByteLevel"},
                "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(vocab_size)}, "merges": []},
            })
            write_json(assets / "tokenizer_config.json", {"tokenizer_class": "GPT2Tokenizer"})
        tensors = self.tensors.copy()
        for name in ("transformer.wte.weight", "lm_head.weight"):
            tensors[name] = (vocab_size, 32), tensors[name][1]
        if indexed:
            entries = list(tensors.items())
            index = {}
            for part in range(2):
                shard = f"model-0000{part + 1}-of-00002.safetensors"
                selection = entries[part::2]
                safetensors(assets / shard, selection)
                index.update(dict.fromkeys((name for name, _ in selection), shard))
            write_json(assets / "model.safetensors.index.json", {"weight_map": index})
            return ["--source", assets], tree_sha256(assets)
        source = self.root / "input.safetensors"
        safetensors(source, tensors.items())
        return ["--source", source, "--assets", assets], sha256(source)

    def invoke(self, operation, flags, *extra):
        return subprocess.run([CLI, operation, *flags, *extra], capture_output=True, text=True, timeout=60)

    def checked(self, operation, flags, *extra):
        result = self.invoke(operation, flags, *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def write_recipe(self, digest, layers=None):
        write_json(self.recipe, {
            "schema": SCHEMA, "source_sha256": digest, "default_type": "BF16",
            "layers": {"0": "F16", "1": "BF16"} if layers is None else layers,
        })

    def assert_types(self, assigned, layer_types=(1, 30)):
        expected = {f"blk.{layer}.{target}.weight": kind
                    for layer, kind in enumerate(layer_types) for target, _ in MATRICES.values()}
        expected.update(dict.fromkeys(self.vectors, 0))
        expected.update({"token_embd.weight": 30, "position_embd.weight": 0, "output.weight": 30})
        self.assertEqual(assigned, expected)

    def test_compile_binds_the_physical_standalone_file(self):
        flags, digest = self.make_source(indexed=False)
        receipt = self.checked("source-digest", flags)
        self.assertEqual(receipt["source_sha256"], digest)
        self.assertEqual(receipt["assets_sha256"], tree_sha256(flags[3]))
        self.write_recipe(digest)
        report = self.checked("compile-layer-recipe", flags, "--recipe", self.recipe)
        self.assert_types(report["expected_ggml_types"])
        self.assertEqual(sha256(flags[1]), digest)
        # Change one tensor payload without changing any header, size, or path.
        with flags[1].open("r+b") as f:
            f.seek(-1, os.SEEK_END)
            tail = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([tail[0] ^ 1]))
        rejected = self.invoke("compile-layer-recipe", flags, "--recipe", self.recipe)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertRegex(rejected.stderr, r"source SHA-256|source_sha256")

    def test_compile_binds_every_file_in_the_indexed_tree(self):
        flags, digest = self.make_source(indexed=True)
        self.write_recipe(digest)
        self.assertEqual(self.checked("source-digest", flags)["source_sha256"], digest)
        report = self.checked("compile-layer-recipe", flags, "--recipe", self.recipe)
        self.assert_types(report["expected_ggml_types"])
        # Whitespace does not change the logical model, but this recipe is
        # bound to the precise physical bytes, including the index itself.
        with (flags[1] / "model.safetensors.index.json").open("ab") as f:
            f.write(b"\n")
        self.assertNotEqual(self.checked("source-digest", flags)["source_sha256"], digest)
        result = self.invoke("compile-layer-recipe", flags, "--recipe", self.recipe)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"source SHA-256|source_sha256")

    def test_export_rejects_an_unmapped_layer_and_never_publishes(self):
        flags, digest = self.make_source(indexed=False)
        self.write_recipe(digest, {"0": "F16", "2": "BF16"})
        output = self.root / "unmapped"
        result = self.invoke("export-layer-recipe", flags, "--recipe", self.recipe, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"unknown.*layer")
        self.assertFalse(output.exists())

    def test_export_rejects_a_bias_that_would_need_silent_widening(self):
        bias = "transformer.h.0.attn.c_attn.bias"
        self.tensors[bias] = (*self.tensors[bias], "F16")
        flags, digest = self.make_source(indexed=False)
        self.write_recipe(digest)
        output = self.root / "widening"
        result = self.invoke("export-layer-recipe", flags, "--recipe", self.recipe, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("blk.0.attn_qkv.bias", result.stderr)
        self.assertFalse(output.exists())

    def test_standalone_never_guesses_adjacent_assets(self):
        flags, _ = self.make_source(indexed=False)
        for asset in flags[3].iterdir():
            shutil.copyfile(asset, self.root / asset.name)
        result = self.invoke("source-digest", flags[:2])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--assets", result.stderr)

    def test_export_does_not_replace_existing_output(self):
        flags, digest = self.make_source(indexed=False)
        self.write_recipe(digest)
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_bytes(b"keep")
        result = self.invoke("export-layer-recipe", flags, "--recipe", self.recipe, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh", result.stderr)
        self.assertEqual(list(output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"keep")

    @unittest.skipUnless(
        all(os.environ.get(key) for key in ("BREWERY_TEST_GPT2_ASSETS", "BREWERY_TEST_CPU_DECODE")),
        "set BREWERY_TEST_GPT2_ASSETS and BREWERY_TEST_CPU_DECODE; run under the CPU resource queue",
    )
    def test_both_sources_emit_exact_types_and_bytes_and_decode_on_cpu(self):
        assets = Path(os.environ["BREWERY_TEST_GPT2_ASSETS"])
        decode = Path(os.environ["BREWERY_TEST_CPU_DECODE"])
        self.assertEqual(sha256(decode), DECODE_SHA256)
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                flags, digest = self.make_source(indexed, assets)
                self.write_recipe(digest, {"0": "Q8_0", "1": "F16"})
                recipe_digest = sha256(self.recipe)
                assets_digest = None if indexed else tree_sha256(flags[3])
                output = self.root / f"export-{indexed}"
                report = self.checked("export-layer-recipe", flags, "--recipe", self.recipe, "--output", output)
                destination = output / "model.gguf"
                self.assertEqual(sha256(destination), report["artifact"]["sha256"])
                actual = read_gguf(destination)
                self.assert_types({name: kind for name, (_, kind, _) in actual.items()}, (8, 1))
                self.assertEqual(
                    report["gguf_inventory"],
                    {name: {"shape": list(shape), "ggml_type": kind} for name, (shape, kind, _) in actual.items()},
                )
                for name, payload in self.vectors.items():
                    self.assertEqual(actual[name][2], payload, name)
                for name, value in self.matrices.items():
                    shape, kind, payload = actual[name]
                    if kind == 8:
                        # A constant positive block stores an F16 scale and 32 int8 127s.
                        block = struct.pack("<e", value / 127) + bytes([127]) * 32
                        expected = block * (math.prod(shape) // 32)
                    else:
                        raw = struct.pack("<f" if kind != 1 else "<e", value)
                        expected = (raw[2:] if kind == 30 else raw) * math.prod(shape)
                    self.assertEqual(payload, expected, name)
                for layer in (0, 1):
                    for target, shape in MATRICES.values():
                        self.assertEqual(actual[f"blk.{layer}.{target}.weight"][0], shape)
                self.assertEqual(tree_sha256(flags[1]) if indexed else sha256(flags[1]), digest)
                if not indexed:
                    self.assertEqual(tree_sha256(flags[3]), assets_digest)
                self.assertEqual(sha256(self.recipe), recipe_digest)
                process = subprocess.run([decode, destination], capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn(f"CPU decode: {GPT2_VOCAB_SIZE} finite logits", process.stdout)


if __name__ == "__main__":
    unittest.main()
