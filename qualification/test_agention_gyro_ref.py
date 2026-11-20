"""Run real reference tests with AGENTION_GYRO_SOURCE at the pinned archive root."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "agention_gyro_ref.py"
REV = "587a5269848eb07996c9c1433f9ad638604eb4d8"
TYPES = {
    "TQ2_T": (144, 34), "TQK6": (145, 26), "TQK7": (146, 30),
    "TQP5": (147, 42), "TQP6": (148, 50), "TQP7": (149, 58),
    "TQK5": (150, 22), "TQK4": (151, 18),
}


def text(s):
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def fixture(path, fields, tensors, alignment=64):
    """Write synthetic GGUF v3 with tuples (name, dims, type_id, bytes)."""
    fields = {"general.alignment": (4, struct.pack("<I", alignment)), **fields}
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(fields))
    for name, (kind, raw) in fields.items():
        header += text(name) + struct.pack("<I", kind) + raw
    data = b""
    for name, dims, kind, raw in tensors:
        header += text(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", kind, len(data))
        data += raw + bytes(-len(raw) % alignment)
    path.write_bytes(header + bytes(-len(header) % alignment) + data)


def weights(i):
    # Deterministic varied rows, independent of the quantizer and decoder.
    return [((j * 29 + i * 43) % 131 - 65) / 97 + math.sin(j / 7) / 8 for j in range(256)]


def command(source, output, upstream, *args, env=None):
    return subprocess.run(
        [sys.executable, str(TOOL), str(source), str(output), "--upstream", str(upstream), *args],
        capture_output=True, text=True, timeout=80, env=env,
    )


@pytest.fixture
def upstream():
    path = os.environ.get("AGENTION_GYRO_SOURCE")
    if not path:
        pytest.skip("set AGENTION_GYRO_SOURCE to the pinned Agention archive root")
    return Path(path)


def reader_at(upstream, path):
    spec = importlib.util.spec_from_file_location("gyro_qualification_decoder", upstream / "python/tq_ref.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GGUF(path)


def test_all_gyro_types_encode_and_read_back_reproducibly(tmp_path, upstream):
    source, out, rerun = [tmp_path / name for name in ("input.gguf", "gyro.gguf", "rerun.gguf")]
    tensors = [(name, (128, 2), 0, struct.pack("<256f", *weights(i)))
               for i, name in enumerate(TYPES)]
    untouched = struct.pack("<4e", -1.5, 0.625, 1.75, -0.25)
    fixture(source, {"general.name": (8, text("synthetic gyro"))},
            [*tensors, ("untouched", (4,), 1, untouched)])
    before = source.read_bytes()
    args = [arg for name in TYPES for arg in ("--tensor", name, name)]
    result = command(source, out, upstream, *args)
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["upstream_revision"] == REV
    assert proof["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert proof["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert len(proof["helper_sha256"]) == 64
    assert proof["importance"] == "none"
    assert source.read_bytes() == before
    assert set(proof["tensors"]) == TYPES.keys()
    for name, (type_id, block_bytes) in TYPES.items():
        row = proof["tensors"][name]
        assert row["type_id"] == type_id
        assert row["packed_bytes"] == 2 * block_bytes
        assert row["independent_decode_match"] is True
        assert 0 < row["relative_l2"] < 0.60

    # Independently read the persisted tensor table and data with the upstream
    # Python reader, rather than accepting the CLI's report alone.
    reader = reader_at(upstream, out)
    assert reader.metadata["general.name"] == "synthetic gyro"
    assert reader.metadata["agention.trellis.state_bits"] == 15
    assert reader.metadata_types["agention.trellis.state_bits"] == (4, None)
    raw = out.read_bytes()
    for (i, (name, (type_id, block_bytes))) in enumerate(TYPES.items()):
        shape, kind, offset = reader.tensors[name]
        assert shape == [128, 2] and kind == type_id
        start = reader.data_offset + offset
        assert any(raw[start:start + 2 * block_bytes])
        following = list(reader.tensors.values())[i + 1][2]
        assert following - offset == (2 * block_bytes + 63) // 64 * 64
        actual = reader.read(name).ravel()
        relative_l2 = math.sqrt(sum((float(x) - y) ** 2 for x, y in zip(actual, weights(i))) /
                                sum(y * y for y in weights(i)))
        assert relative_l2 == pytest.approx(proof["tensors"][name]["relative_l2"], rel=1e-6)
    start = reader.data_offset + reader.tensors["untouched"][2]
    assert raw[start:start + len(untouched)] == untouched

    repeated = command(source, rerun, upstream, *args)
    assert repeated.returncode == 0, repeated.stderr
    assert rerun.read_bytes() == raw


def test_explicit_rotation_preserves_metadata_and_decodes_in_original_basis(tmp_path, upstream):
    source, stored, out, stored_out = (tmp_path / name for name in ("f32", "stored", "gyro", "stored-gyro"))
    name = "blk.0.attn_q.weight"
    signs = [1 if j % 5 in (0, 3) else -1 for j in range(128)]
    words = lambda values: struct.pack("<IQ" + "i" * len(values), 5, len(values), *values)
    metadata = {
        "version": (4, struct.pack("<I", 1)),
        "block_size": (4, struct.pack("<I", 16)),
        "transform": (8, text("normalized-sylvester-walsh-hadamard")),
        "axis": (8, text("input-last-dimension")),
        "sign_mode": (8, text("explicit")),
        "sign_widths": (9, words([128])),
        "sign_values": (9, words(signs)),
        "weight_names": (9, struct.pack("<IQ", 8, 1) + text(name)),
    }
    fields = {f"prism.hadamard.{key}": value for key, value in metadata.items()}
    original = [((j * 17) % 67 - 33) / 128 for j in range(256)]
    # Dense H*S, not the C or Python butterfly. Each factor is exact in F32.
    rotated = [
        sum((-1) ** ((i & j).bit_count()) * original[off + j] * signs[(off + j) % 128] / 4
            for j in range(16))
        for off in range(0, 256, 16) for i in range(16)
    ]
    fixture(source, fields, [(name, (128, 2), 0, struct.pack("<256f", *original))])
    fixture(stored, fields, [(name, (128, 2), 0, struct.pack("<256f", *rotated))])
    args = ("--tensor", name, "TQP7")
    result = command(source, out, upstream, *args)
    assert result.returncode != 0 and "--input-basis" in result.stderr
    assert not out.exists()

    result = command(source, out, upstream, *args, "--input-basis", "original")
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)["tensors"][name]
    assert row["rotation_verified"] is True
    result = command(stored, stored_out, upstream, *args, "--input-basis", "stored")
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == stored_out.read_bytes()
    got = reader_at(upstream, out)
    before = reader_at(upstream, source)
    for field in fields:
        assert got.metadata[field] == before.metadata[field]
        assert got.metadata_types[field] == before.metadata_types[field]
    decoded = got.read(name, unrotate=True).ravel()
    relative_l2 = math.sqrt(sum((float(x) - y) ** 2 for x, y in zip(decoded, original)) /
                            sum(y * y for y in original))
    assert relative_l2 == pytest.approx(row["relative_l2"], rel=1e-6)
    assert relative_l2 < 0.20


def test_wrong_helper_revision_never_publishes(tmp_path, upstream):
    source, out = tmp_path / "input.gguf", tmp_path / "output.gguf"
    fixture(source, {}, [("weights", (128,), 0, struct.pack("<128f", *weights(0)[:128]))])
    # Act as a toolchain that builds a helper for a different upstream pin.
    fake_cc = tmp_path / "cc"
    fake_cc.write_text(
        "#!/bin/sh\n"
        "printf '#!/bin/sh\\necho brewery-agention-gyro-ref-v1 wrong-revision\\n' > helper\n"
        "chmod +x helper\n"
    )
    fake_cc.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    result = command(source, out, upstream, "--tensor", "weights", "TQK4", env=env)
    assert result.returncode != 0 and "helper identity mismatch" in result.stderr
    assert not out.exists()


def test_changed_upstream_source_never_publishes(tmp_path, upstream):
    source, out, modified = (tmp_path / name for name in ("input.gguf", "output.gguf", "modified"))
    fixture(source, {}, [("weights", (128,), 0, struct.pack("<128f", *weights(0)[:128]))])
    (modified / "src").mkdir(parents=True)
    (modified / "src/tq_codebook.c").write_bytes((upstream / "src/tq_codebook.c").read_bytes() + b"\n")
    result = command(source, out, modified, "--tensor", "weights", "TQK4")
    assert result.returncode != 0 and "upstream pin mismatch: src/tq_codebook.c" in result.stderr
    assert not out.exists()


def test_never_clobbers_existing_output_or_input(tmp_path):
    source, out = tmp_path / "input.gguf", tmp_path / "output.gguf"
    fixture(source, {}, [("weights", (128,), 0, struct.pack("<128f", *weights(0)[:128]))])
    out.write_bytes(b"existing artifact")
    for target in (out, source):
        before = target.read_bytes()
        result = command(source, target, tmp_path / "no-upstream", "--tensor", "weights", "TQK4")
        assert result.returncode != 0 and "output already exists" in result.stderr
        assert target.read_bytes() == before


@pytest.mark.parametrize("kind", list(range(144, 152)))
def test_ambiguous_input_type_namespace_is_rejected(tmp_path, kind):
    source, out = tmp_path / "input.gguf", tmp_path / "output.gguf"
    fixture(source, {}, [("weights", (128,), 0, bytes(512)), ("ambiguous", (128,), kind, bytes(64))])
    result = command(source, out, tmp_path / "no-upstream", "--tensor", "weights", "TQK4")
    assert result.returncode != 0 and f"unsupported GGML type {kind}" in result.stderr
    assert not out.exists()


def test_block_budget_rejected_before_compiling(tmp_path):
    source, out = tmp_path / "input.gguf", tmp_path / "output.gguf"
    fixture(source, {}, [("weights", (128, 33), 0, bytes(128 * 33 * 4))])
    result = command(source, out, tmp_path / "no-upstream", "--tensor", "weights", "TQP7")
    assert result.returncode != 0 and "selection exceeds 32 Gyro blocks" in result.stderr
    assert not out.exists()


def test_conflicting_trellis_marker_rejected_before_compiling(tmp_path):
    source, out = tmp_path / "input.gguf", tmp_path / "output.gguf"
    fixture(source, {"agention.trellis.state_bits": (4, struct.pack("<I", 16))},
            [("weights", (128,), 0, bytes(512))])
    result = command(source, out, tmp_path / "no-upstream", "--tensor", "weights", "TQK4")
    assert result.returncode != 0 and "conflicting Agention trellis state width" in result.stderr
    assert not out.exists()
