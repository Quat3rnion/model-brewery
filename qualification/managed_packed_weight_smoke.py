"""Drive packed review/apply through a fresh controller and real CPU worker."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from urllib.error import HTTPError

from managed_weight_smoke import HOST, TENSOR, checksum, controller, request, save, setup

CODECS = {"Q4_0": (2, 32, 16), "Q8_0": (8, 32, 32), "Q8G64": (7, 64, 64)}


def fixture(codec, codes):
    """One actual rank-2 encoded tensor. Both selected cells share a Q4 byte."""
    type_id, width, count = CODECS[codec]
    raw = bytearray([0x88 if codec == "Q4_0" else 0] * count)
    if codec == "Q4_0":
        raw[0] = 8 + codes[0] | ((8 + codes[1]) << 4)
    else:
        raw[0], raw[16] = codes
    scale = struct.pack("<e", 0.5)
    if codec == "Q8G64":
        raw += scale + b"\0\0"
        check = 0
        for (word,) in struct.iter_unpack("<I", raw):
            check ^= word
        header = struct.pack("<4sI4Q64s", b"HGN1", 2, 1, 104, 320, 320 + len(raw), b"packed-fixture")
        table = struct.pack("<96sII4QQQII", TENSOR.encode(), type_id, 2, 1, width, 0, 0,
                            320, len(raw), check, 0)
        return (header + table).ljust(320, b"\0") + raw
    name = TENSOR.encode()
    header = struct.pack("<4sIQQQ", b"GGUF", 3, 1, 0, len(name)) + name
    header += struct.pack("<IQQIQ", 2, width, 1, type_id, 0)
    return header + bytes(-len(header) % 32) + scale + raw


def decode(oracle, codec, path):
    if codec == "Q8G64":
        # Independent affine decode of this one-row fixture; scale .5 and bias
        # 0 are exactly representable. The outer HGN bytes are checked too.
        raw = path.read_bytes()[320:]
        assert len(raw) == 68 and raw[-4:] == b"\x00\x38\x00\x00"
        return struct.pack("<64f", *(code / 2 for code in raw[:64]))
    raw = subprocess.check_output([oracle, path], timeout=10)
    header, values = raw.split(b"\n", 1)
    type_id, _, count, name = header.decode().split()
    assert bytes.fromhex(name).decode() == TENSOR
    assert (int(type_id), int(count), len(values)) == (CODECS[codec][0], 32, 128)
    return values


def run(root, args, codec):
    root.mkdir(mode=0o700)
    inputs = setup(root, args.bin_dir.resolve(strict=True), args.gguf_python.resolve(strict=True))
    fmt = "hgn" if codec == "Q8G64" else "gguf"
    for location in ("worker.json", "dispatch.json"):
        cfg = root / location
        data = json.loads(cfg.read_bytes())
        if location == "worker.json":
            data["engine_env"].update(BREWERY_WEIGHT_PACKED="0" if args.no_runtime else "1",
                                      BREWERY_WEIGHT_HGN="1")
        else:
            data["hosts"][HOST]["capabilities"].update(weight_packed=not args.no_cap, weight_hgn=True)
        save(cfg, data)
    source = inputs / f"source.{fmt}"
    original = fixture(codec, (0, 0))
    source.write_bytes(original)
    pin = checksum(source)
    width = CODECS[codec][1]
    first, second = [int(i == 0) for i in range(width)], [int(i == 16) for i in range(width)]
    save(inputs / "cal.json", {"activations": [first, second], "targets": [[1], [1.5]]})
    save(inputs / "held.json", {"activations": [[a + b for a, b in zip(first, second)]], "targets": [[0]]})

    with controller(root, 0) as url:
        assert url.startswith("http://127.0.0.1:") and not url.endswith(":4141")

        def http(route, body=None):
            return request(url, route, body)[0]

        def register(path, kind):
            return http("models", {"name": path.name, "host": HOST, "kind": kind, "path": str(path)})["id"]

        def payload(kind, ids, options):
            return {"kind": kind, "host": HOST, "input_ids": ids, "options": options}

        def submit(label, req, expected="succeeded"):
            try:
                job_id = http("jobs", req)["id"]
            except HTTPError as error:
                save(root / f"{label}.json", {"status": error.code, "error": error.read().decode()})
                assert error.code == expected or (expected == "legacy-refusal" and error.code in (400, 409)), label
                return error.code
            assert not isinstance(expected, int), (label, "unexpectedly accepted", job_id)
            deadline = time.monotonic() + 20
            while True:
                job = http(f"jobs/{job_id}")
                if job["state"] in ("failed", "succeeded", "cancelled"):
                    break
                assert time.monotonic() < deadline, (label, "job timed out")
                time.sleep(0.1)
            save(root / f"{label}.json", job)
            assert job["state"] == ("failed" if expected == "legacy-refusal" else expected), (label, job)
            return job

        state = http("state")
        assert not state["models"] and not state["jobs"]
        cap = state["hosts"][0]["capabilities"].get("weight_packed", False)
        assert cap == (not (args.expect_old or args.no_cap)), state["hosts"]
        sid = register(source, fmt)
        inventory_req = payload("weight_inspect", [sid],
                                {"mode": "inventory", "offset": 0, "limit": 4, "packed": True})
        if args.expect_old or args.no_cap:
            denied = submit("packed-refused", inventory_req, 409 if args.no_cap else "legacy-refusal")
            assert len(http("state")["models"]) == 1 and source.read_bytes() == original
            assert not list((root / "worker").glob("jobs/*/work/edited.*"))
            return {"codec": codec, "refused": denied if isinstance(denied, int) else denied["state"]}
        if args.no_runtime:
            denied = submit("packed-runtime-refused", inventory_req, "failed")
            assert "packed" in str(denied.get("error", "")).lower(), denied
            assert len(http("state")["models"]) == 1 and source.read_bytes() == original
            assert not list((root / "worker").glob("jobs/*/work/edited.*"))
            return {"codec": codec, "refused": "worker runtime gate"}

        inventory = submit("inventory", inventory_req)["result"]
        assert inventory["packed"] is True
        row, = inventory["report"]["tensors"]
        assert (row["name"], row["dtype"], row["shape"], row["editable"]) == (TENSOR, codec, [1, width], True)
        assert inventory["report"]["source_sha256"] == pin
        tile = submit("tile", payload("weight_inspect", [sid], {
            "mode": "tile", "tensor": TENSOR, "origin": [0, 0], "shape": [1, 17],
            "source_sha256": pin, "packed": True,
        }))["result"]
        assert tile["packed"] is True and tile["report"]["values"] == [[0] * 17]
        for idx in (0, 16):
            assert tile["report"]["packed_steps"][0][idx]["upper"] == 0.5
        manual = submit("manual-review", payload("weight_propose", [sid], {
            "packed": True, "source_sha256": pin, "edits": [
                {"tensor": TENSOR, "index": [0, c], "op": "set", "value": value}
                for c, value in ((0, 0.74), (16, 1.24))],
        }))
        ids = [register(inputs / f"{name}.json", "weight-calibration") for name in ("cal", "held")]
        automatic = submit("tune-review", payload("weight_tune", [sid, *ids], {
            "packed": True, "source_sha256": pin, "tensor": TENSOR, "indices": [[0, 0], [0, 16]],
            "max_update": 1.5,
        }))
        tune = automatic["result"]["report"]["tune"]
        assert tune["calibration"] == {"before_mse": 1.625, "after_mse": 0}
        assert tune["held_out"] == {"before_mse": 0, "after_mse": 6.25, "regression": True}
        assert len(http("state")["models"]) == 3 and source.read_bytes() == original
        submit("unreviewed", payload("weight_apply", [sid],
               {"proposal_job_id": manual["id"], "reviewed": False}), 400)
        old_values = decode(args.oracle, codec, source)
        artifacts = {}
        for label, review, codes in (("manual", manual, (1, 2)), ("tune", automatic, (2, 3))):
            result = review["result"]
            assert result["packed"] is True
            proposal = result["report"]["proposal"]
            assert proposal["source_sha256"] == pin
            assert [(e["dtype"], e["index"], e["before"], e["after"]) for e in proposal["edits"]] == [
                (codec, [0, 0], 0, codes[0] / 2), (codec, [0, 16], 0, codes[1] / 2)]
            final = submit(label + "-apply", payload("weight_apply", [sid],
                           {"proposal_job_id": review["id"], "reviewed": True}))
            result = final["result"]
            assert result["verified"] is True and result["packed"] is True
            assert result["source_sha256"] == pin and result["report"]["output_sha256"] == result["artifact"]["sha256"]
            model = next(m for m in http("state")["models"] if m["id"] == result["library_model_id"])
            assert (model["kind"], model["host"]) == (fmt, HOST)
            path = Path(model["path"])
            assert path == root / "worker/jobs" / final["id"] / f"work/edited.{fmt}"
            assert path.read_bytes() == fixture(codec, codes)
            expected = bytearray(old_values)
            struct.pack_into("<f", expected, 0, codes[0] / 2)
            struct.pack_into("<f", expected, 16 * 4, codes[1] / 2)
            assert decode(args.oracle, codec, path) == expected
            changes = [i for i, (a, b) in enumerate(zip(original, path.read_bytes())) if a != b]
            artifacts[label] = {"path": str(path), "sha256": checksum(path), "changed_byte_offsets": changes}
        assert source.read_bytes() == original and len(http("state")["models"]) == 5
        return {"codec": codec, "source": str(source), "source_sha256": pin, "outputs": artifacts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bin-dir", "gguf-python", "oracle"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--codec", choices=CODECS, action="append")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--expect-old", action="store_true")
    modes.add_argument("--no-cap", action="store_true")
    modes.add_argument("--no-runtime", action="store_true")
    args = parser.parse_args()
    for key in tuple(os.environ):
        if key.startswith("BREWERY_"):
            del os.environ[key]
    os.umask(0o077)
    os.environ["TMPDIR"] = "/arboretum"
    root = Path(tempfile.mkdtemp(prefix="managed-packed-weight.", dir="/arboretum"))
    print(f"Scratch evidence: {root}", flush=True)
    report = [run(root / codec, args, codec) for codec in (args.codec or CODECS)]
    save(root / "summary.json", report)
    print(json.dumps(report))
