"""Bounded real HTTP smoke for standalone SafeTensors and indexed HF edits."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError

from managed_weight_smoke import HOST, TENSOR, controller, matrix, request, save, setup

FLOAT = "layer.float"
INTEGER = "layer.indices"
SHARD = "model-00001-of-00002.safetensors"
SECOND = "model-00002-of-00002.safetensors"
INDEX = "model.safetensors.index.json"
ORIGINAL = (1, 2, 3, 4)
PROOF_ASSET = "proof-fixture.bin"
PROOF_ASSET_BYTES = (64 << 20) + 1


def bf16(values):
    # All fixture values are exactly representable BF16.
    return b"".join(struct.pack("<f", value)[2:] for value in values)


def safetensors(*entries):
    header = {"__metadata__": {"format": "pt", "fixture": "independent"}}
    data = bytearray()
    for name, dtype, shape, value in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(value)]}
        data += value
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + data


def fixture(kind, values=ORIGINAL):
    weight = (TENSOR, "BF16", [2, 2], bf16(values))
    floating = (FLOAT, "F32", [1, 2], struct.pack("<2f", 1.25, -2.5))
    integer = (INTEGER, "I64", [2], struct.pack("<2q", -3, (1 << 54) + 1))
    if kind == "safetensors":
        return {"": safetensors(floating, weight, integer)}
    return {
        SHARD: safetensors(weight),
        SECOND: safetensors(integer, floating),
        INDEX: json.dumps({"metadata": {"total_size": 32}, "weight_map": {
            INTEGER: SECOND, TENSOR: SHARD, FLOAT: SECOND,
        }}, indent=2).encode() + b"\n",
        "config.json": b'{ "architectures": ["Fixture"], "hidden_size": 2 }\n',
        "tokenizer.json": b'{ "version": "1.0" }\n',
        "extra.txt": b"Keep exact bytes in flat sidecars.\n",
    }


def write_fixture(path, files):
    for name, data in files.items():
        dest = path / name if name else path
        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        dest.write_bytes(data)


def describe(path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if not p.is_dir())
    snapshot = {}
    for file in files:
        info = file.stat()
        assert not file.is_symlink() and file.is_file() and info.st_nlink == 1, file
        name = file.relative_to(path).as_posix() if path.is_dir() else ""
        assert info.st_size < 8192 or (name == PROOF_ASSET and info.st_size == PROOF_ASSET_BYTES), file
        snapshot[name] = (file.read_bytes(), (info.st_dev, info.st_ino))
    if path.is_dir():
        assert not any(p.is_dir() for p in path.rglob("*"))
    return snapshot


def digest(files):
    if "" in files:
        return sha256(files[""]).hexdigest()
    return sha256("".join(f"{name}\0{len(data)}\0{sha256(data).hexdigest()}\n"
                          for name, data in sorted(files.items())).encode()).hexdigest()


def standard_read(path, kind):
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    paths = [path] if kind == "safetensors" else [path / SHARD, path / SECOND]
    seen = {}
    for item in paths:
        with safe_open(item, framework="numpy") as reader:
            for name in reader.keys():
                view = reader.get_slice(name)
                seen[name] = (view.get_dtype(), view.get_shape())
    assert seen == {TENSOR: ("BF16", [2, 2]), FLOAT: ("F32", [1, 2]), INTEGER: ("I64", [2])}
    return True


def run(root, args, kind):
    root.mkdir(mode=0o700)
    bins = args.bin_dir.resolve(strict=True)
    inputs = setup(root, bins, args.gguf_python.resolve(strict=True))
    for name in ("dispatch.json", "worker.json"):
        config = root / name
        data = json.loads(config.read_bytes())
        if name == "dispatch.json":
            data["hosts"][HOST]["capabilities"]["weight_safetensors"] = not args.no_cap
        else:
            data["engine_env"].update(BREWERY_WEIGHT_SAFETENSORS="1",
                                      BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES="67108864")
        save(config, data)
    source = inputs / ("source.safetensors" if kind == "safetensors" else "source-hf")
    other = inputs / ("wrong.safetensors" if kind == "safetensors" else "wrong-hf")
    original = fixture(kind)
    write_fixture(source, original)
    write_fixture(other, fixture(kind, (9, 2, 3, 4)))
    if args.packaged_proof:
        for helper in ("edit", "tune"):
            launcher = bins / f"brewery-weight-{helper}"
            capability = json.loads(subprocess.check_output([launcher, "--capabilities"], timeout=15))
            assert capability["proof_source_fds"] is True
        with (source / PROOF_ASSET).open("xb") as asset:
            asset.truncate(PROOF_ASSET_BYTES)
        original[PROOF_ASSET] = bytes(PROOF_ASSET_BYTES)
    pin = digest(original)
    source_before = describe(source)
    index_pin = None
    if args.packaged_proof:
        receipt, chunk_index = root / "fixture-publication.json", root / "fixture-chunk-index.json"
        save(receipt, {"schema": "model-brewery.source-publication.v1", "model_dir": str(source),
                       "files": {n: {"size": len(data), "sha256": sha256(data).hexdigest()}
                                 for n, data in sorted(original.items())}})
        built = subprocess.check_output([bins / "brewery-worker", "build-weight-chunk-index",
                                         receipt, pin, chunk_index], timeout=20)
        index_pin = json.loads(built)["ok"]["sha256"]
        assert index_pin == sha256(chunk_index.read_bytes()).hexdigest()
        worker = root / "worker.json"
        config = json.loads(worker.read_bytes())
        config["engine_env"].update(BREWERY_WEIGHT_CHUNK_INDEX=str(chunk_index),
                                    BREWERY_WEIGHT_CHUNK_INDEX_SHA256=index_pin)
        save(worker, config)
    save(inputs / "cal.json", {"activations": [[1, 0], [0, 1]], "targets": [[2, 3], [2, 3]]})
    save(inputs / "held.json", {"activations": [[1, 1], [2, -1]], "targets": [[3, 7], [0, 2]]})

    with controller(root, 0) as url:
        assert url.startswith("http://127.0.0.1:") and not url.endswith(":4141"), url

        def http(route, body=None):
            return request(url, route, body)[0]

        def register(path, fmt):
            return http("models", {"name": path.name, "path": str(path), "kind": fmt, "host": HOST})["id"]

        def post(job_kind, ids, options):
            return http("jobs", {"kind": job_kind, "host": HOST, "input_ids": ids, "options": options})

        def refuse(label, action, statuses):
            current = http("state")
            try:
                action()
            except HTTPError as error:
                response = error.read().decode()
                save(root / f"{label}.json", {"status": error.code, "response": response})
                assert error.code in statuses, (label, error.code, response)
                assert http("state") == current, label
                return error.code
            raise AssertionError(f"{label}: accepted")

        def job(label, job_kind, ids, options, expected="succeeded"):
            ident = post(job_kind, ids, options)["id"]
            deadline = time.monotonic() + 20
            while True:
                row = http(f"jobs/{ident}")
                if row["state"] not in {"queued", "running"}:
                    break
                assert time.monotonic() < deadline, (label, ident)
                time.sleep(0.1)
            save(root / f"{label}.json", row)
            assert row["state"] == expected, (label, row)
            return row

        state = http("state")
        cap = state["hosts"][0]["capabilities"].get("weight_safetensors", False)
        assert not state["models"] and not state["jobs"]
        assert cap == (not (args.expect_old or args.no_cap)), state["hosts"]
        if args.expect_old and kind == "safetensors":
            status = refuse("old-register-safetensors", lambda: register(source, kind), (400,))
            return {"format": kind, "refusal": status}
        source_id = register(source, kind)
        inventory_opts = {"mode": "inventory", "offset": 0, "limit": 16}
        if args.expect_old or args.no_cap:
            status = refuse("unsupported-worker", lambda: post("weight_inspect", [source_id], inventory_opts),
                            (400, 409) if args.expect_old else (409,))
            return {"format": kind, "refusal": status}
        other_id = register(other, kind)
        cal, held = [register(inputs / f"{name}.json", "weight-calibration") for name in ("cal", "held")]
        initial_models = http("state")["models"]
        refuse("packed-rejected", lambda: post("weight_inspect", [source_id], {**inventory_opts, "packed": True}),
               (400,))

        def inspect(label, ident, opts, expected_sha):
            result = job(label, "weight_inspect", [ident], opts)["result"]
            assert result["source_format"] == kind and result["report"]["source_sha256"] == expected_sha
            return result["report"]

        inventory = inspect("inventory", source_id, inventory_opts, pin)
        tensors = {row["name"]: row for row in inventory["tensors"]}
        assert inventory["total"] == 3 and set(tensors) == {TENSOR, FLOAT, INTEGER}
        assert [(tensors[n]["dtype"], tensors[n]["shape"]) for n in (TENSOR, FLOAT, INTEGER)] == [
            ("BF16", [2, 2]), ("F32", [1, 2]), ("I64", [2])]
        assert tensors[TENSOR]["editable"] is True and tensors[INTEGER]["editable"] is False

        for label, invalid_name in (("over-limit", "x" * 96), ("metadata", "__metadata__"),
                                    ("control", "tensor\n"), ("non-ascii", "ténsor")):
            refuse(f"native-name-{label}", lambda: post(
                "weight_inspect", [source_id], {"tensor": invalid_name, "index": [0, 0]}), (400,))
        legacy = inputs / "legacy.gguf"
        legacy.write_bytes(matrix(ORIGINAL))
        legacy_id = register(legacy, "gguf")
        refuse("gguf-still-rejects-long-name", lambda: post(
            "weight_inspect", [legacy_id], {"tensor": "x" * 64, "index": [0, 0]}), (400,))
        initial_models = http("state")["models"]

        def tile(label, ident, expected_sha, values):
            result = inspect(label, ident, {"mode": "tile", "tensor": TENSOR, "origin": [0, 0],
                                           "shape": [2, 2], "source_sha256": expected_sha}, expected_sha)
            assert result["tensor"] == TENSOR and result["values"] == [list(values[:2]), list(values[2:])], result

        tile("source-tile", source_id, pin, ORIGINAL)
        scalar = inspect("source-scalar", source_id, {"tensor": TENSOR, "index": [0, 1]}, pin)
        assert scalar["tensor"] == TENSOR and scalar["value"] == 2
        manual = job("manual-proposal", "weight_propose", [source_id], {"source_sha256": pin, "edits": [
            {"tensor": TENSOR, "index": [0, 1], "op": "delta", "value": 0.5}]})
        auto = job("tune-proposal", "weight_tune", [source_id, cal, held], {
            "source_sha256": pin, "tensor": TENSOR, "indices": [[0, 0], [1, 1]], "max_update": 1})
        metrics = auto["result"]["report"]["tune"]
        assert metrics["calibration"] == {"before_mse": 0.5, "after_mse": 0}
        assert metrics["held_out"] == {"before_mse": 0, "after_mse": 1.75, "regression": True}
        assert http("state")["models"] == initial_models and describe(source) == source_before
        assert not list((root / "worker").glob("jobs/*/work/edited.*"))
        approve = {"proposal_job_id": manual["id"], "reviewed": True}
        refuse("wrong-source", lambda: post("weight_apply", [other_id], approve), (400,))
        refuse("unreviewed", lambda: post("weight_apply", [source_id], {**approve, "reviewed": False}), (400,))

        if args.packaged_proof:
            asset = source / PROOF_ASSET
            with asset.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                stream.write(b"X")
            tile("untouched-sidecar-not-read", source_id, pin, ORIGINAL)
            failed = job("corrupt-sidecar-apply", "weight_apply", [source_id], approve, expected="failed")
            assert not (root / "worker/jobs" / failed["id"] / "work/edited.hf").exists()
            assert http("state")["models"] == initial_models
            with asset.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                stream.write(b"\0")

        outputs = {}
        for label, review, edits, values in (
            ("manual", manual, [([0, 1], 2, 2.5)], (1, 2.5, 3, 4)),
            ("auto", auto, [([0, 0], 1, 2), ([1, 1], 4, 3)], (2, 2, 3, 3)),
        ):
            proposal = review["result"]["report"]["proposal"]
            assert review["result"]["source_format"] == kind
            assert proposal["schema"] == "brewery.weight-edit-proposal.v1" and proposal["source_sha256"] == pin
            assert [(e["index"], e["before"], e["after"]) for e in proposal["edits"]] == edits
            assert all(e["dtype"] == "BF16" and e["tensor"] == TENSOR for e in proposal["edits"])
            final = job(f"{label}-apply", "weight_apply", [source_id],
                        {"proposal_job_id": review["id"], "reviewed": True})
            result = final["result"]
            assert result["verified"] is True and result["source_format"] == kind
            assert result["source_sha256"] == pin
            assert result["proposal_sha256"] == review["result"]["artifact"]["sha256"]
            model = next(m for m in http("state")["models"] if m["id"] == result["library_model_id"])
            output = Path(model["path"])
            assert model["host"] == HOST and model["kind"] == kind
            assert output == root / "worker/jobs" / final["id"] / f"work/edited.{kind}"
            actual = describe(output)
            expected = fixture(kind, values)
            if args.packaged_proof:
                expected[PROOF_ASSET] = original[PROOF_ASSET]
            assert {name: data for name, (data, _) in actual.items()} == expected
            assert not {inode for _, inode in actual.values()} & {inode for _, inode in source_before.values()}
            output_pin = digest(expected)
            artifact = result["artifact"]
            assert result["report"]["output_sha256"] == output_pin == artifact["sha256"]
            if kind == "hf":
                assert artifact["bundle_sha256"] == output_pin
                assert artifact["files"] == [
                    {"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
                    for name, data in sorted(expected.items())]
            assert describe(source) == source_before
            if not args.packaged_proof:
                # Scaled outputs need their own independent proof before sparse reopening.
                tile(f"{label}-reuse-tile", model["id"], output_pin, values)
            outputs[label] = {"path": str(output), "sha256": output_pin,
                              "standard_reader": standard_read(output, kind), "job_id": final["id"]}
        assert len(http("state")["models"]) == len(initial_models) + 2
        summary = {"format": kind, "source": str(source), "source_sha256": pin, "outputs": outputs}
        if args.packaged_proof:
            summary.update(chunk_index_sha256=index_pin, source_bytes=sum(map(len, original.values())),
                           untouched_corruption_rejected=True,
                           launch_sha256={n: sha256((bins / n).read_bytes()).hexdigest() for n in (
                               "brewery", "brewery-dispatch", "brewery-worker", "brewery-engine",
                               "brewery-weight-edit", "brewery-weight-tune")})
        return summary


if __name__ == "__main__":
    # The production inventory listed this exact 72-byte BF16 key; the
    # previous tile request validator rejected it before dispatch.
    TENSOR = "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--gguf-python", required=True, type=Path)
    parser.add_argument("--format", choices=("safetensors", "hf"), action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--expect-old", action="store_true")
    mode.add_argument("--no-cap", action="store_true")
    mode.add_argument("--packaged-proof", action="store_true",
                      help="Run scaled indexed HF with the real offline package and worker chunk proof")
    args = parser.parse_args()
    if args.packaged_proof and args.format not in (None, ["hf"]):
        parser.error("--packaged-proof requires indexed HF")
    for key in tuple(os.environ):
        if key.startswith("BREWERY_"):
            del os.environ[key]
    os.umask(0o077)
    os.environ["TMPDIR"] = "/arboretum"
    root = Path(tempfile.mkdtemp(prefix="managed-safetensors.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    formats = args.format or (("hf",) if args.packaged_proof else ("safetensors", "hf"))
    report = [run(root / kind, args, kind) for kind in formats]
    save(root / "summary.json", report)
    print(json.dumps(report))
