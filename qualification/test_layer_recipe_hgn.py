"""Exercise the offline recipe CLI, not the model runtime."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from qualification.test_layer_recipe_gguf import read_gguf


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/layer_recipe_hgn.py"
HELPER = os.environ.get("BREWERY_LAYER_HGN_HELPER")
QWEN2_PARITY = os.environ.get("BREWERY_LAYER_QWEN2_PARITY")
RECEIPT = "layer-recipe.receipt.json"
QWEN2_LAYER_NAMES = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run_export(source, recipe, output, *extra, helper="not-called"):
    return subprocess.run(
        [sys.executable, str(CLI), str(source), str(recipe), str(output),
         "--helper", str(helper), *map(str, extra)],
        capture_output=True, text=True,
    )


def write_safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype, "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def llama_fixture(assets, *, trellis=False):
    assets.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 64, "intermediate_size": 64, "num_attention_heads": 2,
        "num_key_value_heads": 1, "num_hidden_layers": 3, "vocab_size": 16,
        "max_position_embeddings": 128, "tie_word_embeddings": False,
    }
    if trellis:
        config.update(hidden_size=128, intermediate_size=256, num_key_value_heads=2)
    (assets / "config.json").write_text(json.dumps(config))
    (assets / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, **{
            f"t{i}": i for i in range(1, 16)}}, "unk_token": "[UNK]"},
    }))
    tensors, expected = {}, {}

    def add(name, shape, values, target):
        # Test data contains BF16-exact values. F16 and Q8G64 expected bytes
        # are built independently of either converter or HGN writer.
        raw = b"".join(struct.pack("<f", x)[2:] for x in values)
        tensors[name] = ("BF16", shape, raw)
        if target == "F16":
            output = struct.pack(f"<{len(values)}e", *values)
        elif target == "Q8G64":
            # Each row contains 0..62,255 (so scale=1 and bias=0).
            output = (bytes(range(63)) + bytes.fromhex("ff 00 3c 00 00")) * shape[0]
        else:
            output = raw
        expected[name] = (target, shape, output)

    if trellis:
        for name, shape in (("model.embed_tokens.weight", [16, 128]),
                            ("lm_head.weight", [16, 128]), ("model.norm.weight", [128])):
            add(name, shape, [-0.5, 1.25] * (math.prod(shape) // 2), "BF16")
        for layer in range(3):
            for i, (suffix, shape) in enumerate((
                ("self_attn.q_proj", [128, 128]), ("self_attn.k_proj", [128, 128]),
                ("self_attn.v_proj", [128, 128]), ("self_attn.o_proj", [128, 128]),
                ("mlp.gate_proj", [256, 128]), ("mlp.up_proj", [256, 128]),
                ("mlp.down_proj", [128, 256]),
            )):
                # Known, BF16-exact weights; row 0 is dead, others have varying RMS.
                values = [0.0 if r == 0 else
                          (((r * 7 + c * 5 + (r ^ c) + layer * 11 + i * 19) % 37) - 18)
                          * (1 + r % 4) / 64
                          for r in range(shape[0]) for c in range(shape[1])]
                add(f"model.layers.{layer}.{suffix}.weight", shape, values, "BF16")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                add(f"model.layers.{layer}.{norm}.weight", [128], [1.25] * 128, "BF16")
        return tensors, expected

    add("model.embed_tokens.weight", [16, 64], [1.5] * 1024, "BF16")
    add("model.norm.weight", [64], [0.75] * 64, "BF16")
    add("lm_head.weight", [16, 64], [-1.5] * 1024, "BF16")
    for layer, target in enumerate(("F16", "BF16", "Q8G64")):
        prefix = f"model.layers.{layer}."
        for i, suffix in enumerate((
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )):
            rows = 32 if suffix in ("self_attn.k_proj", "self_attn.v_proj") else 64
            if target == "Q8G64":
                values = (list(range(63)) + [255]) * rows
            else:
                values = ([i + 0.125, i - 0.5, -0.0, 0.25] * 16) * rows
            add(prefix + suffix + ".weight", [rows, 64], values, target)
        add(prefix + "input_layernorm.weight", [64], [1.25] * 64, "BF16")
        add(prefix + "post_attention_layernorm.weight", [64], [0.5] * 64, "BF16")
    return tensors, expected


def qwen2_fixture(assets):
    tensors, _ = llama_fixture(assets)
    path = assets / "config.json"
    config = json.loads(path.read_bytes())
    config.update(architectures=["Qwen2ForCausalLM"], model_type="qwen2", num_hidden_layers=2)
    path.write_text(json.dumps(config))
    tensors = {name: row for name, row in tensors.items() if not name.startswith("model.layers.2.")}
    for name, (_, shape, raw) in list(tensors.items()):
        if len(shape) == 1:
            tensors[name] = ("F32", shape, b"".join(b"\0\0" + raw[i:i + 2] for i in range(0, len(raw), 2)))
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            tensors[name.removesuffix("weight") + "bias"] = (
                "F32", shape[:1], struct.pack("<f", 0.3125) * shape[0],
            )
    return tensors


def snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def indexed_digest(files):
    return sha(b"".join(
        f"{name}\0{len(data)}\0{sha(data)}\n".encode()
        for name, data in sorted(files.items())
    ))


def read_hgn_v2(raw):
    """Independent HGN 1.1.0 header/table reader; no converter code or receipt."""
    magic, version, count, table, data, size = struct.unpack_from("<4sIQQQQ", raw)
    if (magic, version, table, size) != (b"HGN1", 2, 104, len(raw)) or data % 64:
        raise ValueError("invalid HGN header")
    tensors, minimum = {}, data
    for at in range(table, table + 160 * count, 160):
        name, dtype, rank, *fields = struct.unpack_from("<96sII4qQQII", raw, at)
        name = name.rstrip(b"\0").decode("ascii")
        shape, offset, length, checksum, variant = fields[:rank], *fields[4:]
        if offset < minimum or offset % 64 or offset + length > size or name in tensors:
            raise ValueError("invalid HGN tensor table")
        minimum = offset + length
        payload = raw[offset:minimum]
        xor32 = 0
        for word, in struct.iter_unpack("<I", payload + b"\0" * (-length % 4)):
            xor32 ^= word
        if xor32 != checksum:
            raise ValueError(f"HGN checksum mismatch: {name}")
        tensors[name] = dtype, variant, shape, payload, offset
    return tensors


def decode_trellis(payload, rows, cols, bits, su, sv):
    """Decode circular states and invert the Hadamards from STORAGE.md, not Rust."""
    matrix = [[0.0] * cols for _ in range(rows)]
    limit = 256 * bits
    tile = 0
    for row_band in range(0, rows, 128):
        for col in range(0, cols, 16):
            for row in range(row_band, row_band + 128, 16):
                words = struct.unpack_from(f"<{8 * bits}I", payload, tile)
                tile += 32 * bits
                stream = "".join(f"{word:032b}" for word in words)
                cyclic = stream[limit - 16:] + stream
                for i in range(256):
                    end = bits * (i + 1)
                    state = int(cyclic[end:end + 16], 2)
                    a = sum(((state * 0x83DCD12D) & 0xFFFFFFFF).to_bytes(4, "little"))
                    value, = struct.unpack("<e", struct.pack("<e", (1774 * a - 905216) / 2**18))
                    matrix[row + (i // 8) % 16][col + 8 * (i // 128) + i % 8] = value

    def butterfly(v):
        for step in (1, 2, 4, 8, 16, 32, 64):
            for start in range(0, 128, 2 * step):
                for i in range(start, start + step):
                    a, b = v[i], v[i + step]
                    v[i], v[i + step] = a + b, a - b
        return v

    for n in range(0, rows, 128):
        for k in range(0, cols, 128):
            block = [butterfly(row[k:k + 128]) for row in matrix[n:n + 128]]
            for c, col in enumerate(zip(*block)):
                for r, value in enumerate(butterfly(list(col))):
                    matrix[n + r][k + c] = (value / 128) * su[k + c] * sv[n + r]
    return matrix


class LayerRecipeHgnTests(unittest.TestCase):
    @unittest.skipUnless(HELPER, "set BREWERY_LAYER_HGN_HELPER to the built upstream native adapter")
    def test_trellis_overrides_write_real_native_payloads(self):
        tmp = os.environ.get("TMPDIR", "/arboretum")
        with tempfile.TemporaryDirectory(dir=tmp) as work:
            root = Path(work)
            assets = root / "assets"
            tensors, _ = llama_fixture(assets, trellis=True)
            source, recipe, output = root / "model.safetensors", root / "recipe.json", root / "bundle"
            write_safetensors(source, tensors)
            original = source.read_bytes()
            recipe.write_text(json.dumps({
                "schema": "brewery.layer-quant.v1", "source_sha256": sha(original),
                "default_type": "BF16", "layers": {"0": "HT3", "1": "HT4"},
            }))
            result = run_export(source, recipe, output, "--assets", assets, helper=HELPER)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = output / "model.hgn"
            file_bytes = model.read_bytes()
            actual = read_hgn_v2(file_bytes)
            receipt = json.loads((output / RECEIPT).read_bytes())
            records = {row["name"]: row for row in receipt["tensors"]}
            self.assertTrue(receipt["structural_reader_verified"])
            self.assertFalse(receipt["runtime_load_verified"])
            self.assertEqual(receipt["model"]["sha256"], sha(file_bytes))
            self.assertEqual(receipt["trellis"]["spec_commit"], "79e75fd4ae7642478823525b0a3212951f584af5")
            self.assertRegex(receipt["trellis"]["encoder_sha256"], r"^[0-9a-f]{64}$")
            expected_names = set(tensors)
            first = "model.layers.0.self_attn.q_proj.weight"
            for name, (_, shape, raw) in tensors.items():
                with self.subTest(tensor=name):
                    dtype, variant, actual_shape, payload, _ = actual[name]
                    self.assertEqual(actual_shape, shape)
                    self.assertEqual(records[name]["source_payload_sha256"], sha(raw))
                    self.assertEqual(records[name]["native_payload_sha256"], sha(payload))
                    if len(shape) != 2 or not name.startswith(("model.layers.0.", "model.layers.1.")):
                        self.assertEqual((dtype, variant, payload), (0, 0, raw))
                        continue
                    rows, cols = shape
                    bits = 3 if name.startswith("model.layers.0.") else 4
                    self.assertEqual((dtype, variant, len(payload)), (16, 0x1200 + 2 * bits, rows * cols * bits // 8))
                    self.assertEqual(records[name]["native_type"], f"HT{bits}")
                    self.assertGreater(len(set(payload)), 100)
                    vectors = []
                    for suffix, length in (("suh", cols), ("svh", rows)):
                        companion = name.removesuffix("weight") + suffix
                        expected_names.add(companion)
                        vtype, vvariant, vshape, vraw, _ = actual[companion]
                        self.assertEqual((vtype, vvariant, vshape, len(vraw)), (2, 0, [length], 2 * length))
                        self.assertEqual(records[name]["rotation"][suffix], {
                            "name": companion, "shape": [length], "native_type": "F16",
                            "native_payload_sha256": sha(vraw),
                        })
                        vectors.append(struct.unpack(f"<{length}e", vraw))
                    su, sv = vectors
                    self.assertEqual(set(su), {-1.0, 1.0})
                    source_values = [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0]
                                     for i in range(0, len(raw), 2)]
                    for row, scale in enumerate(sv):
                        rms = math.sqrt(math.fsum(x * x for x in source_values[row * cols:(row + 1) * cols]) / cols)
                        self.assertEqual(abs(scale), struct.unpack("<e", struct.pack("<e", rms))[0])
                    decoded = decode_trellis(payload, rows, cols, bits, su, sv)
                    self.assertEqual(decoded[0], [0.0] * cols)
                    error = math.fsum((x - y)**2 for x, y in
                                      zip(source_values, (v for row in decoded for v in row)))
                    energy = math.fsum(x * x for x in source_values)
                    self.assertLess(math.sqrt(error / energy), 0.18 if bits == 3 else 0.11)
            self.assertEqual(set(actual), expected_names)
            self.assertEqual(source.read_bytes(), original)
            repeated = root / "repeat"
            result = run_export(source, recipe, repeated, "--assets", assets, helper=HELPER)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((repeated / "model.hgn").read_bytes(), file_bytes)
            corrupted = bytearray(file_bytes)
            corrupted[actual[first][4] + 43] ^= 1
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_hgn_v2(corrupted)
            corrupt = root / "corrupt.hgn"
            corrupt.write_bytes(corrupted)
            inspect = subprocess.run([HELPER, "inspect", corrupt], capture_output=True, text=True)
            self.assertNotEqual(inspect.returncode, 0)
            self.assertIn("checksum mismatch", inspect.stderr)

    def test_trellis_rejects_misaligned_gqa_rows_and_nonlayer_types(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/arboretum")) as work:
            root = Path(work)
            assets = root / "assets"
            tensors, _ = llama_fixture(assets, trellis=True)
            config = assets / "config.json"
            config.write_text(json.dumps({**json.loads(config.read_bytes()), "num_key_value_heads": 1}))
            for name, (dtype, shape, raw) in list(tensors.items()):
                if name.endswith(("k_proj.weight", "v_proj.weight")):
                    tensors[name] = (dtype, [shape[0] // 2, shape[1]], raw[:len(raw) // 2])
            source, recipe, output = root / "model.safetensors", root / "recipe.json", root / "bundle"
            write_safetensors(source, tensors)
            for layers, default, message in (
                ({"0": "HT3"}, "BF16", "HT3 requires rows and columns divisible by 128: model.layers.0.self_attn.k_proj.weight"),
                ({"0": "HT4"}, "BF16", "HT4 requires rows and columns divisible by 128: model.layers.0.self_attn.k_proj.weight"),
                ({}, "HT4", "layer overrides only"),
                ({"0": "HT43"}, "BF16", "HT43 is a profile, not a codec"),
                ({"0": "I4R"}, "BF16", "requests 'I4R'"),
            ):
                with self.subTest(layers=layers, default=default):
                    recipe.write_text(json.dumps({
                        "schema": "brewery.layer-quant.v1", "source_sha256": sha(source.read_bytes()),
                        "default_type": default, "layers": layers,
                    }))
                    result = run_export(source, recipe, output, "--assets", assets)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())

    def test_qwen2_rejects_narrow_vectors_and_missing_bias(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as work:
            root = Path(work)
            assets = root / "assets"
            tensors = qwen2_fixture(assets)
            source, recipe, output = root / "model.safetensors", root / "recipe.json", root / "bundle"
            bias = "model.layers.1.self_attn.k_proj.bias"
            _, shape, _ = tensors[bias]
            for weights, message in (
                ({**tensors, bias: ("BF16", shape, b"\x80\x3f" * shape[0])}, "requires source F32 vectors"),
                ({name: row for name, row in tensors.items() if name != bias}, "missing="),
            ):
                with self.subTest(message=message):
                    write_safetensors(source, weights)
                    recipe.write_text(json.dumps({"schema": "brewery.layer-quant.v1",
                                                  "source_sha256": sha(source.read_bytes()),
                                                  "layers": {}, "default_type": "F16"}))
                    result = run_export(source, recipe, output, "--assets", assets)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertIn(bias, result.stderr)
                    self.assertFalse(output.exists())

    @unittest.skipUnless(HELPER, "set BREWERY_LAYER_HGN_HELPER to the built upstream native adapter")
    def test_qwen2_default_converts_global_and_unoverridden_matrices(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as work:
            root = Path(work)
            assets = root / "assets"
            tensors = qwen2_fixture(assets)
            source, recipe, output = root / "model.safetensors", root / "recipe.json", root / "bundle"
            write_safetensors(source, tensors)
            recipe.write_text(json.dumps({"schema": "brewery.layer-quant.v1",
                                          "source_sha256": sha(source.read_bytes()),
                                          "layers": {"0": "BF16"}, "default_type": "F16"}))
            result = run_export(source, recipe, output, "--assets", assets, helper=HELPER)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((output / RECEIPT).read_bytes())
            dt = {row["name"]: row["native_type"] for row in receipt["tensors"]}
            self.assertEqual(dt, {
                name: "F32" if len(shape) == 1 else "BF16" if name.startswith("model.layers.0.") else "F16"
                for name, (_, shape, _) in tensors.items()
            })

    @unittest.skipUnless(HELPER and QWEN2_PARITY, "set the native adapter and paired Qwen2 fixture paths")
    def test_qwen2_matches_all_shared_gguf_payloads(self):
        fixture = Path(QWEN2_PARITY)
        source, recipe = fixture / "model.safetensors", fixture / "recipe.json"
        assets = fixture.parent / "paired/assets"
        gguf = read_gguf(fixture / "gguf-out/model.gguf")
        names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                 "lm_head.weight": "output.weight"}
        names.update({f"model.layers.{layer}.{src}": f"blk.{layer}.{dest}"
                      for layer in range(2) for src, dest in QWEN2_LAYER_NAMES.items()})
        before, before_assets = source.read_bytes(), snapshot(assets)
        header_len, = struct.unpack_from("<Q", before)
        original = json.loads(before[8:8 + header_len])
        original.pop("__metadata__", None)
        self.assertEqual(set(names), set(original))
        self.assertEqual(set(names.values()), set(gguf))
        self.assertEqual(len(names), 27)
        with tempfile.TemporaryDirectory(dir="/arboretum") as work:
            bundle = Path(work) / "bundle"
            result = run_export(source, recipe, bundle, "--assets", assets, helper=HELPER)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = bundle / "model.hgn"
            reader = json.loads(subprocess.check_output([HELPER, "inspect", model]))
            self.assertEqual(reader["version"], 2)
            hgn = {row["name"]: row for row in reader["tensors"]}
            self.assertEqual(set(names), set(hgn))
            self.assertEqual(len(reader["tensors"]), 27)
            hgn_bytes = model.read_bytes()
            for src, dest in names.items():
                with self.subTest(tensor=src):
                    shape, kind, actual_gguf = gguf[dest]
                    row, src_meta = hgn[src], original[src]
                    lo, hi = src_meta["data_offsets"]
                    exact_source = before[8 + header_len + lo:8 + header_len + hi]
                    dtype = "F32" if len(shape) == 1 else "F16" if dest.startswith("blk.0.") else "BF16"
                    self.assertEqual(kind, {"F32": 0, "F16": 1, "BF16": 30}[dtype])
                    self.assertEqual((row["dtype"], row["shape"]), (dtype, list(reversed(shape))))
                    self.assertEqual(row["shape"], src_meta["shape"])
                    expected = exact_source
                    if dtype == "F16":
                        expected = b"".join(struct.pack("<e", struct.unpack_from(
                            "<f", b"\0\0" + exact_source[i:i + 2])[0])
                            for i in range(0, len(exact_source), 2))
                    # Qwen2 GGUF keeps HF row order, including the Q/K weights and biases.
                    self.assertEqual(actual_gguf, expected)
                    self.assertEqual(hgn_bytes[row["offset"]:row["offset"] + row["size"]], expected)
                    self.assertEqual(row["sha256"], sha(expected))
            receipt = json.loads((bundle / RECEIPT).read_bytes())
            self.assertEqual(receipt["architecture"], "Qwen2ForCausalLM")
            self.assertFalse(receipt["runtime_load_verified"])
            self.assertEqual(receipt["asset_sha256"], {name: sha(data) for name, data in before_assets.items()})
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(snapshot(assets), before_assets)

    def test_recipe_is_bound_to_physical_source_before_publication(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as work:
            root = Path(work)
            source = root / "model.safetensors"
            write_safetensors(source, {"x": ("BF16", [1], b"\x80\x3f")})
            before = source.read_bytes()
            recipe, output = root / "recipe.json", root / "bundle"
            recipe.write_text(json.dumps({
                "schema": "brewery.layer-quant.v1",
                "source_sha256": "0" * 64,
                "layers": {"0": "F16"},
            }))
            result = run_export(source, recipe, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source SHA-256 mismatch", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), before)

    def test_incomplete_or_unsupported_inputs_leave_no_bundle(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as work:
            root = Path(work)
            assets = root / "assets"
            tensors, _ = llama_fixture(assets)
            source, recipe, output = root / "model.safetensors", root / "recipe.json", root / "bundle"
            valid = {"0": "F16", "1": "BF16", "2": "BF16"}
            for weights, layers, use_assets, expected in (
                (tensors, valid, False, "requires --assets"),
                (tensors, {**valid, "0": "Q8_0"}, True, "native HGN types"),
                (tensors, {**valid, "0": "F32"}, True, "widening is not supported"),
                (tensors, {"0": "F16"}, True, "missing layer assignment: 1"),
                ({name: t for name, t in tensors.items() if name != "lm_head.weight"},
                 valid, True, "incomplete or unsupported Llama tensor topology"),
                ({**tensors, "unexpected.weight": ("BF16", [1], b"\x80\x3f")},
                 valid, True, "incomplete or unsupported Llama tensor topology"),
            ):
                with self.subTest(expected=expected):
                    write_safetensors(source, weights)
                    recipe.write_text(json.dumps({
                        "schema": "brewery.layer-quant.v1", "source_sha256": sha(source.read_bytes()),
                        "layers": layers,
                    }))
                    result = run_export(source, recipe, output, *(("--assets", assets) if use_assets else ()))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse(output.exists())
            write_safetensors(source, tensors)
            recipe.write_text(json.dumps({
                "schema": "brewery.layer-quant.v1", "source_sha256": sha(source.read_bytes()),
                "layers": valid,
            }))
            tokenizer = assets / "tokenizer.json"
            broken = json.loads(tokenizer.read_bytes())
            broken["model"]["vocab"]["bad"] = 42
            tokenizer.write_text(json.dumps(broken))
            result = run_export(source, recipe, output, "--assets", assets)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tokenizer vocabulary IDs exceed Llama embeddings", result.stderr)
            self.assertFalse(output.exists())

    @unittest.skipUnless(HELPER, "set BREWERY_LAYER_HGN_HELPER to the built upstream native adapter")
    def test_real_v2_reader_verifies_each_standalone_and_indexed_payload(self):
        for indexed in (False, True):
            with self.subTest(indexed=indexed), tempfile.TemporaryDirectory(dir="/arboretum") as work:
                root = Path(work)
                assets = root / ("source" if indexed else "assets")
                tensors, expected = llama_fixture(assets)
                if indexed:
                    source = assets
                    shards = ("model-00001.safetensors", "model-00002.safetensors")
                    weights = list(tensors)
                    for offset, shard in enumerate(shards):
                        write_safetensors(source / shard, {
                            name: tensors[name] for name in weights[offset::2]
                        })
                    (source / "model.safetensors.index.json").write_text(json.dumps({
                        "weight_map": {name: shards[i % 2] for i, name in enumerate(weights)},
                        "metadata": {"total_size": sum(len(row[2]) for row in tensors.values())},
                    }))
                    before = snapshot(source)
                    digest = indexed_digest(before)
                    extra = ()
                else:
                    source = root / "model.safetensors"
                    write_safetensors(source, tensors)
                    before = source.read_bytes()
                    digest = sha(before)
                    extra = ("--assets", assets)
                recipe, output = root / "recipe.json", root / "bundle"
                recipe.write_text(json.dumps({
                    "schema": "brewery.layer-quant.v1", "source_sha256": digest,
                    "default_type": "BF16", "layers": {"0": "F16", "2": "Q8G64"},
                }))
                result = run_export(source, recipe, output, *extra, helper=HELPER)
                self.assertEqual(result.returncode, 0, result.stderr)
                model = output / "model.hgn"
                inventory = json.loads(subprocess.check_output([HELPER, "inspect", model]))
                self.assertEqual(inventory["version"], 2)
                self.assertEqual(len(inventory["tensors"]), len(tensors))
                self.assertSetEqual({row["name"] for row in inventory["tensors"]}, set(tensors))
                receipt = json.loads((output / RECEIPT).read_bytes())
                self.assertTrue(receipt["structural_reader_verified"])
                self.assertFalse(receipt["runtime_load_verified"])
                self.assertIn("Llama", receipt["runtime_blocker"])
                self.assertEqual(receipt["source_sha256"], digest)
                model_bytes = model.read_bytes()
                self.assertEqual(receipt["model"]["sha256"], sha(model_bytes))
                by_name = {row["name"]: row for row in receipt["tensors"]}
                for row in inventory["tensors"]:
                    name = row["name"]
                    dtype, shape, raw = expected[name]
                    self.assertEqual((row["dtype"], row["shape"], row["size"]),
                                     (dtype, shape, len(raw)), name)
                    self.assertEqual(row["sha256"], sha(raw), name)
                    self.assertEqual(model_bytes[row["offset"]:row["offset"] + row["size"]], raw, name)
                    self.assertEqual(by_name[name]["source_payload_sha256"], sha(tensors[name][2]), name)
                    self.assertEqual(by_name[name]["native_payload_sha256"], row["sha256"], name)
                for asset in ("config.json", "tokenizer.json"):
                    original = (assets / asset).read_bytes()
                    self.assertEqual((output / asset).read_bytes(), original)
                    self.assertEqual(receipt["asset_sha256"][asset], sha(original))
                output_before = snapshot(output)
                again = run_export(source, recipe, output, *extra, helper=HELPER)
                self.assertNotEqual(again.returncode, 0)
                self.assertIn("destination already exists", again.stderr)
                self.assertEqual(snapshot(output), output_before)
                self.assertEqual(snapshot(source) if indexed else source.read_bytes(), before)
