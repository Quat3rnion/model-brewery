"""Public CLI regressions for SafeTensors weight revisions."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CHUNK = 1 << 20


def tensor_file(name, dtype, payload, shape):
    head = json.dumps({name: {"dtype": dtype, "shape": shape, "data_offsets": [0, len(payload)]},
                       "__metadata__": {"format": "pt"}}).encode()
    head += b" " * (-len(head) % 8)
    return struct.pack("<Q", len(head)) + head + payload


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin_dir(path):
    return sha(b"".join(
        f"{f.relative_to(path).as_posix()}\0{f.stat().st_size}\0{sha(f.read_bytes())}\n".encode()
        for f in sorted(path.rglob("*"), key=lambda f: f.relative_to(path).as_posix()) if f.is_file()
    ))


class SafeTensors(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir="/arboretum", prefix="weight-safetensors-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def cli(self, *args, tool="weight_edit.py", ok=True):
        result = subprocess.run([sys.executable, "-B", TOOLS / tool, *map(str, args)],
                                capture_output=True, text=True, timeout=15)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def doc(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def chunk_manifest(self, materials):
        manifest = self.doc("chunks", [
            {"path": name, "bytes": len(raw), "sha256": sha(raw),
             "chunk_sha256": [sha(raw[i:i + CHUNK]) for i in range(0, len(raw), CHUNK)]}
            for name, raw in sorted(materials.items())
        ])
        manifest.chmod(0o600)
        return manifest

    def test_chunk_manifest_reads_cross_chunk_tile_and_rejects_corruption(self):
        source = self.root / "source.safetensors"
        raw = tensor_file("layer.weight", "F32", struct.pack("<2f", 3., 4.), [1, 2])
        head = raw[8:-8]
        # Put the two selected cells on opposite sides of the chunk boundary.
        raw = struct.pack("<Q", CHUNK - 12) + head.ljust(CHUNK - 12, b" ") + raw[-8:]
        source.write_bytes(raw)
        manifest = self.chunk_manifest({source.name: raw})
        args = ("--source-sha256", sha(raw), "--source-manifest", manifest)
        tile = ("tile", source, *args, "--tensor", "layer.weight", "--origin", "[0,0]",
                "--shape", "[1,2]")
        self.assertEqual(self.cli(*tile)["values"], [[3., 4.]])
        # Tune consumes the entire matrix in one range across the boundary.
        cal = self.doc("calibration", {"activations": [[1, 0], [0, 1]], "targets": [[4], [4]]})
        held = self.doc("held-out", {"activations": [[1, 1]], "targets": [[8]]})
        tune = (source, *args, "--tensor", "layer.weight", "--indices", "[[0,0]]",
                "--max-update", 1, "--calibration", cal, "--held-out", held)
        self.assertEqual(self.cli(*tune, tool="weight_tune.py")["edits"][0]["after"], 4.)
        source.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        # The second chunk is not needed for inventory but is needed for the tile.
        self.assertEqual(self.cli("inventory", source, *args, "--offset", 0, "--limit", 1)
                         ["tensors"][0]["name"], "layer.weight")
        self.assertIn("chunk SHA-256 mismatch", self.cli(*tile, ok=False))
        self.assertIn("chunk SHA-256 mismatch", self.cli(*tune, tool="weight_tune.py", ok=False))

    def test_chunk_manifest_authenticates_index_and_apply_still_hashes_unread_sidecars(self):
        source, original = self.indexed()
        manifest = self.chunk_manifest(original)
        args = ("--source-sha256", pin_dir(source), "--source-manifest", manifest)
        inventory = ("inventory", source, *args, "--offset", 0, "--limit", 10)
        index_name = "model.safetensors.index.json"
        index = source / index_name
        # This is still valid JSON with exactly the same mapping and length.
        index.write_bytes(original[index_name].replace(b": ", b":\t"))
        self.assertIn(f"chunk SHA-256 mismatch: {index_name}", self.cli(*inventory, ok=False))
        index.write_bytes(original[index_name])
        self.assertEqual(self.cli(*inventory)["total"], 2)

        request = self.doc("request", [{"tensor": "layer.weight", "index": [0, 0],
                                       "op": "set", "value": 2.0}])
        approved = self.doc("approved", self.cli("propose", source, *args, request))
        out = self.root / "revision"
        apply = ("apply", source, "--source-manifest", manifest, approved, out)
        sidecar = source / "engram.dat"
        sidecar.write_bytes(b"Exact sidecar")
        self.assertIn("source changed since verified manifest", self.cli(*apply, ok=False))
        self.assertFalse(out.exists())
        sidecar.write_bytes(original[sidecar.name])
        self.assertEqual(self.cli(*apply)["output_sha256"], pin_dir(out))
        self.assertEqual((out / sidecar.name).read_bytes(), original[sidecar.name])

    def test_chunk_manifest_rejects_wrong_count_bad_hex_and_mixed_member_proofs(self):
        source = self.root / "source.safetensors"
        raw = tensor_file("layer.weight", "F32", struct.pack("<f", 3.), [1])
        source.write_bytes(raw)
        manifest = self.chunk_manifest({source.name: raw})
        valid, = json.loads(manifest.read_bytes())
        inventory = ("inventory", source, "--source-sha256", sha(raw), "--source-manifest",
                     manifest, "--offset", 0, "--limit", 1)
        for field in ([], [sha(raw)] * 2, [sha(raw).upper()], [1], None):
            with self.subTest(chunks=field):
                manifest.write_text(json.dumps([{**valid, "chunk_sha256": field}]))
                self.assertIn("invalid source chunk SHA-256 list", self.cli(*inventory, ok=False))
        st = source.stat()
        manifest.write_text(json.dumps([{**valid, "device": st.st_dev, "inode": st.st_ino}]))
        self.assertIn("invalid source manifest entry", self.cli(*inventory, ok=False))

    def test_standalone_preserves_all_other_bytes_and_dtype(self):
        source = self.root / "source.safetensors"
        original = tensor_file("layer.weight", "F16", struct.pack("<4e", 1, 2, 3, 4), [2, 2])
        source.write_bytes(original)
        pin = ("--source-sha256", sha(original))
        tile = self.cli("tile", source, *pin, "--tensor", "layer.weight",
                        "--origin", "[1,0]", "--shape", "[1,2]")
        self.assertEqual(tile["source_format"], "safetensors")
        self.assertEqual(tile["values"], [[3., 4.]])
        request = self.doc("request", [{"tensor": "layer.weight", "index": [1, 0],
                                        "op": "set", "value": 1.0009}])
        proposal = self.cli("propose", source, *pin, request)
        self.assertEqual(proposal["edits"][0]["after"], 1.0009765625)
        approval = self.doc("approved", proposal)
        out = self.root / "out.safetensors"
        result = self.cli("apply", source, approval, out)
        self.assertEqual(out.read_bytes(), original[:-4] + b"\x01\x3c" + original[-2:])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(result["output_sha256"], sha(out.read_bytes()))
        self.cli("apply", source, approval, out, ok=False)

    def indexed(self):
        source = self.root / "checkpoint"
        source.mkdir()
        a = tensor_file("layer.weight", "BF16", struct.pack("<4H", 0x3f80, 0x4000, 0x4040, 0x4080), [2, 2])
        b = tensor_file("visual.weight", "F32", struct.pack("<2f", 5., 6.), [1, 2])
        materials = {
            "model-00001-of-00002.safetensors": a,
            "model-00002-of-00002.safetensors": b,
            "model.safetensors.index.json": json.dumps({
                "weight_map": {"layer.weight": "model-00001-of-00002.safetensors",
                               "visual.weight": "model-00002-of-00002.safetensors"},
                "metadata": {"total_size": 16.0},
            }).encode(),
            "config.json": b'{"model_type":"fixture"}\n',
            "tokenizer.json": b'{"version":"1.0"}\n',
            "vision/processor.json": b'{}\n',
            "mtp/overlay.safetensors": tensor_file("mtp.weight", "F32", struct.pack("<f", 7.), [1]),
            "engram.dat": b"exact sidecar",
        }
        for name, data in materials.items():
            f = source / name
            f.parent.mkdir(exist_ok=True)
            f.write_bytes(data)
        return source, materials

    def test_indexed_tune_apply_preserves_complete_tree(self):
        source, original = self.indexed()
        pin = pin_dir(source)
        manifest = self.doc("manifest", [
            {"path": name, "bytes": len(raw), "sha256": sha(raw)}
            for name, raw in sorted(original.items())
        ])
        manifest.chmod(0o600)
        args = ("--source-sha256", pin, "--source-manifest", manifest)
        inventory = self.cli("inventory", source, *args, "--offset", 0, "--limit", 10)
        self.assertEqual(inventory["source_format"], "hf")
        self.assertEqual([t["name"] for t in inventory["tensors"]], ["layer.weight", "visual.weight"])
        cal = self.doc("calibration", {"activations": [[1, 0], [0, 1]],
                                       "targets": [[2, 3], [2, 3]]})
        held = self.doc("held-out", {"activations": [[1, 1], [2, -1]],
                                      "targets": [[3, 7], [0, 2]]})
        proposal = self.cli(source, *args, "--tensor", "layer.weight",
                            "--indices", "[[0,0],[1,1]]", "--max-update", 1,
                            "--calibration", cal, "--held-out", held, tool="weight_tune.py")
        self.assertEqual([e["after"] for e in proposal["edits"]], [2., 3.])
        out = self.root / "revision"
        result = self.cli("apply", source, "--source-manifest", manifest,
                          self.doc("approved", proposal), out)
        self.assertEqual(result["output_sha256"], pin_dir(out))
        self.assertEqual(result["source_format"], "hf")
        changed = dict(original)
        shard = "model-00001-of-00002.safetensors"
        changed[shard] = original[shard][:-8] + struct.pack("<4H", 0x4000, 0x4000, 0x4040, 0x4040)
        self.assertEqual({p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()},
                         set(original))
        for name in original:
            self.assertEqual((source / name).read_bytes(), original[name])
            self.assertEqual((out / name).read_bytes(), changed[name])
            self.assertNotEqual((out / name).stat().st_ino, (source / name).stat().st_ino)

    def test_invalid_mapping_links_and_changed_shard_fail_closed(self):
        source, original = self.indexed()
        shard = "model-00001-of-00002.safetensors"
        original_sha = pin_dir(source)
        options = ("--offset", 0, "--limit", 10)
        link = source / "substitution"
        link.symlink_to(source / shard)
        self.assertIn("without links", self.cli("inventory", source, "--source-sha256",
                                               original_sha, *options, ok=False))
        link.unlink()
        os.link(source / shard, link)
        self.assertIn("without links", self.cli("inventory", source, "--source-sha256",
                                               original_sha, *options, ok=False))
        link.unlink()
        index = source / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer.weight": "../outside.safetensors"}}))
        self.assertIn("unsafe indexed", self.cli("inventory", source, "--source-sha256",
                                               pin_dir(source), *options, ok=False))
        index.write_bytes(original[index.name])
        manifest = self.doc("manifest", [
            {"path": name, "bytes": len(raw), "sha256": sha(raw)}
            for name, raw in sorted(original.items())
        ])
        manifest.chmod(0o600)
        old = original[shard]
        (source / shard).write_bytes(old[:-1] + bytes([old[-1] ^ 1]))
        self.assertIn("mismatch", self.cli("inventory", source, "--source-sha256",
                     original_sha, "--source-manifest", manifest, *options, ok=False))

    def test_noncanonical_tensor_extent_and_unsupported_edit_are_refused(self):
        source = self.root / "malformed.safetensors"
        broken = tensor_file("layer.weight", "F32", b"xx", [1, 1])
        source.write_bytes(broken)
        self.assertIn("payload size mismatch", self.cli("inventory", source, "--source-sha256",
                      sha(broken), "--offset", 0, "--limit", 10, ok=False))
        raw = tensor_file("count", "I64", struct.pack("<q", 42), [])
        source.write_bytes(raw)
        pin = ("--source-sha256", sha(raw))
        row, = self.cli("inventory", source, *pin, "--offset", 0, "--limit", 10)["tensors"]
        self.assertEqual(row["shape"], [])
        self.assertFalse(row["editable"])
        self.assertIn("read-only", self.cli("inspect", source, *pin,
                                           "--tensor", "count", "--index", "[]", ok=False))

    def test_large_unproven_source_rejected_before_hashing_payload(self):
        source = self.root / "unproven.safetensors"
        with source.open("wb") as out:
            out.truncate((64 << 20) + 1)
        self.assertIn("requires operator-verified immutable inode manifest", self.cli(
            "inventory", source, "--source-sha256", "0" * 64, "--offset", 0,
            "--limit", 1, ok=False))
        # A claimed immutable proof must first match the actual open inode;
        # user-authored inode numbers cannot bypass even this metadata check.
        actual = source.stat()
        manifest = self.doc("inode-proof", [{"path": "original.safetensors",
                            "bytes": actual.st_size, "sha256": "0" * 64,
                            "device": actual.st_dev, "inode": actual.st_ino + 1}])
        manifest.chmod(0o600)
        self.assertIn("size or inode differs", self.cli(
            "inventory", source, "--source-sha256", "0" * 64, "--source-manifest",
            manifest, "--offset", 0, "--limit", 1, ok=False))


if __name__ == "__main__":
    unittest.main()
