#!/usr/bin/env python3
"""Offline exact-export smoke on the 65 pinned llama-quantize fixtures.

Use the pinned llama.cpp gguf-py via PYTHONPATH and NumPy. Pass the output
directory from gguf_catalog_smoke.py as --fixtures, and a new scratch path
under /arboretum as --out. This tests tensor bytes and CPU decoding, not
model execution or GPU support.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gguf

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "crates/controller/src/gguf_catalog.json"
EXACT_EXPORT = ROOT / "tools/exact-export"
sys.path.insert(0, str(ROOT / "tools/cross_container"))
from ggml_in_hgn import MAX_FILE, Reader  # noqa: E402


def need(condition, why):
    if not condition:
        raise ValueError(why)


def file_digest(path):
    need(path.is_file() and not path.is_symlink(), "expected a regular fixture/export")
    need(0 < path.stat().st_size <= MAX_FILE, "fixture/export exceeds offline size limit")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_case(src, dst, case):
    sha = file_digest(src)
    result = subprocess.run(
        [sys.executable, "-B", str(EXACT_EXPORT), "--encoded-sha256", sha,
         "--to", "hgn", str(src), str(dst)],
        capture_output=True, text=True, timeout=30,
    )
    need(result.returncode == 0, result.stderr.strip())
    receipt = json.loads(result.stdout)
    need(receipt["encoded_sha256"] == sha and receipt["output_sha256"] == file_digest(dst)
         and receipt["source_format"] == "gguf" and receipt["format"] == "hgn"
         and receipt["verification"] == "adapter-verified" and receipt["runtime_tested"] is False,
         "wrong exact-export receipt")

    original = gguf.GGUFReader(src)
    shelf = Reader(dst)
    fields = list(original.fields.values())[3:]  # exclude synthetic GGUF header fields
    need(list(shelf.metadata) == [field.name for field in fields], "GGUF metadata keys changed")
    for field in fields:
        actual = shelf.metadata[field.name]
        need(actual.gguf_type == int(field.types[0]) and actual.raw == b"".join(
            part.tobytes() for part in field.parts), f"GGUF metadata changed: {field.name}")
    need(shelf.source_sha256 == sha, "wrong GGUF source identity in shelf")
    need(list(shelf.tensors) == [tensor.name for tensor in original.tensors],
         "tensor membership or order changed")
    need(set(case["tensors"]) == set(shelf.tensors), "fixture inventory differs from encoder report")

    seen = Counter()
    no_cpu = set()
    elements = 0
    encoded_bytes = 0
    for t in original.tensors:
        kind = t.tensor_type
        shape = tuple(int(n) for n in t.shape[::-1])
        expected = t.data.tobytes()
        expected_record = case["tensors"][t.name]
        need(expected_record["shape"] == list(t.shape)
             and expected_record["type_id"] == int(kind)
             and expected_record["sha256"] == hashlib.sha256(expected).hexdigest(),
             f"fixture tensor differs from encoder report: {t.name}")
        m = shelf.tensors[t.name]
        need(m.ggml_type == int(kind) and m.shape == shape and m.size == len(expected)
             and shelf.encoded(t.name) == expected, f"HGN tensor/encoded bytes changed: {t.name}")
        seen[kind.name] += 1
        encoded_bytes += len(expected)
        try:
            reference = gguf.dequantize(t.data, kind)
        except NotImplementedError:
            no_cpu.add(kind.name)
            continue
        actual = shelf.decode(t.name)
        need(actual.shape == reference.shape and actual.dtype == reference.dtype
             and actual.tobytes() == reference.tobytes(), f"CPU decode differs: {t.name}")
        elements += int(reference.size)
    return {"tensors": dict(seen), "encoded_bytes": encoded_bytes,
            "cpu_values": elements, "without_cpu_decode": sorted(no_cpu)}


def smoke(fixtures, out):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    want_cases = {f"global-{p['id']}" for p in catalog["presets"]}
    want_cases |= {f"exact-{t['id']}" for t in catalog["tensor_types"]}
    want_cases.add("global-MXFP4_MOE-experts")
    report = fixtures / "report.json"
    need(report.is_file() and not report.is_symlink() and report.stat().st_size <= 2 << 20,
         "missing or oversized encoder fixture report")
    source_report = json.loads(report.read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in source_report["runs"]}
    need(source_report["schema"] == "brewery-gguf-catalog-smoke-v1"
         and source_report["catalog_revision"] == catalog["encoder_revision"]
         and len(cases) == len(want_cases) == len(source_report["runs"]) == 65
         and set(cases) == want_cases and source_report["failed"] == 0
         and all(case["status"] == "passed" for case in cases.values()),
         "expected all 65 successful pinned encoder fixtures")
    rows = []
    types = Counter()
    without_cpu = set()
    for name in sorted(cases):
        try:
            details = check_case(fixtures / f"{name}.gguf", out / f"{name}.hgn", cases[name])
            types.update(details["tensors"])
            without_cpu.update(details["without_cpu_decode"])
            rows.append({"case": name, "status": "passed", **details})
        except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
            rows.append({"case": name, "status": "failed", "error": str(exc)})
            print(f"failed {name}: {exc}", file=sys.stderr)
    missing = sorted({t["id"] for t in catalog["tensor_types"]} - types.keys())
    passed = sum(row["status"] == "passed" for row in rows)
    result = {
        "schema": "brewery-ggml-hgn-catalog-smoke-v1",
        "encoder_revision": catalog["encoder_revision"],
        "encoder_binary_sha256": source_report["encoder_binary_sha256"],
        "passed": passed, "failed": len(rows) - passed,
        "tensor_count": sum(types.values()), "encoding_counts": dict(sorted(types.items())),
        "encoded_bytes": sum(r.get("encoded_bytes", 0) for r in rows),
        "cpu_values": sum(r.get("cpu_values", 0) for r in rows),
        "without_cpu_decode": sorted(without_cpu), "missing_catalog_encodings": missing,
        "cases": rows,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))
    return int(passed != len(rows) or bool(missing))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new scratch directory")
    args = ap.parse_args()
    try:
        out = args.out.resolve()
        out.mkdir(mode=0o700)
        return smoke(args.fixtures, out)
    except (ValueError, OSError, KeyError) as exc:
        print(f"ggml_hgn_catalog_smoke: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
