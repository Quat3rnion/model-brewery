"""Dedicated Brewery worker engine: one pinned BF16 gate -> GGUF Q8_0."""
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import struct
import sys

spec = importlib.util.spec_from_file_location("hy4_plan", Path(__file__).with_name("hy4_plan.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
require = helpers.require

TARGET = "blk.1.attn_gate.weight"
SOURCE = "model.layers.1.self_attn.linear_gate.weight"
SHARD = "model-00065-of-00148.safetensors"
PINS = {
    "control": "96dca0d8d9075201fa18999591900e1377b7a2c4ea904774845cee8e215ba0ba",
    "preflight": "07435a454f2e92be58f6c0d2452ad4d0e6bd7ce8bc549c39d3c60fefcab214c2",
    "quantizer": "e295e66700f1e871521e9ebab7d6a6722fa81b3ea41baf77af14202ffd445c16",
    "model.safetensors.index.json": "0cd31d2ffceb989e4d465d0c80be5731f6941dbf434ad84506e30930fa02e192",
    "config.json": "a59ea10f98683e83c6515e340c85e24eb2e48c91d75af491ed3d6e88c3bc1b3c",
}
MAX_FILE = 224299813888
CHUNK = 16 << 20
OLD = helpers.OLD
NEW = b"Q8_0 block1 attn_gate; Q4_K experts; raw".ljust(len(OLD))


def read(fd, n, at):
    data = os.pread(fd, n, at)
    require(len(data) == n, "Short Hy4 input or readback")
    return data


def write(fd, data, at):
    require(os.pwrite(fd, data, at) == len(data), "Short Hy4 output write")


def chunks(start, length):
    for p in range(start, start + length, CHUNK):
        yield p, min(CHUNK, start + length - p)


def pinned(root, name, digest, limit):
    with helpers.member(root, name) as (f, size):
        require(size <= limit, "Hy4 metadata/codec exceeds its bound")
        data = read(f.fileno(), size, 0)
        require(helpers.sha(data) == digest, f"{name}: pinned SHA-256 mismatch")
        return data


def layout(header, common, tensors, disk, size):
    """Decode the actual header, checking every tensor and span against preflight."""
    f = io.BytesIO(header)
    require(f.read(4) == b"GGUF", "Not GGUF")
    version, count, kv = struct.unpack("<IQQ", f.read(20))
    require(version == 3 and count == len(tensors) and kv <= 64, "Unexpected Hy4 GGUF header")
    slots = {}
    for _ in range(kv):
        key = common.read_gguf_string(f, "metadata key")
        qtype = common.read_u32(f, "metadata type")
        at = f.tell()
        common.skip_gguf_value(f, qtype)
        require(key not in slots and f.tell() <= len(header), "Invalid Hy4 metadata extent")
        slots[key] = (qtype, at, f.tell())
    expected = {
        "general.architecture": (8, b"hy_v4_flash"),
        "halo.hy4.quantization": (8, None),
        "halo.hy4.engram.file_offset": (10, struct.pack("<Q", disk)),
        "halo.hy4.engram.disk_bytes": (10, struct.pack("<Q", size - disk)),
        "general.alignment": (4, struct.pack("<I", 32)),
        "halo.hy4.artifact_version": (4, struct.pack("<I", 1)),
        "halo.hy4.engram.row_bytes": (4, struct.pack("<I", 192)),
    }
    for key, (t, val) in expected.items():
        actual, a, b = slots[key]
        require(actual == t and (val is None or header[a:b] ==
                ((struct.pack("<Q", len(val)) if t == 8 else b"") + val)),
                f"Unexpected Hy4 metadata: {key}")
    last = len(header)
    fields, tables = [], []
    for t in tensors:
        name = common.read_gguf_string(f, "tensor name")
        rank = common.read_u32(f, "rank")
        require(1 <= rank <= 4, "Invalid tensor rank")
        shape = [common.read_u64(f, "dimension") for _ in range(rank)]
        at = f.tell()
        qtype, offset = struct.unpack("<IQ", f.read(12))
        fields.append(at)
        require((name, shape, qtype, offset) == (t["name"], t["shape"], t["qtype"], t["offset"]),
                f"Control/preflight descriptor mismatch: {name}")
        amount = 4 * common.product(shape) if qtype == 26 else common.qtype_nbytes(qtype, shape)
        start = len(header) + offset
        require(amount == t["nbytes"] and start >= last and start % 32 == 0,
                f"Invalid Hy4 span: {name}")
        if t["role"] == "engram_disk":
            require(start == (last if tables else disk) and qtype == 30 and shape[0] == 96,
                    "Invalid BF16 Engram extent")
            tables.append(dict(name=name, qtype=qtype, rows=shape[1], file_offset=start, bytes=amount))
        else:
            require(not tables and start == last and start + amount <= disk, "Invalid resident span")
        last = start + common.align(amount)
    require(tables and disk % 65536 == 0 and last == size and
            0 <= len(header) - f.tell() < 32 and not any(f.read()), "Invalid Hy4 tail/header padding")
    return slots, fields, tables


def source_gate(root, common, pins, shape):
    index = json.loads(pinned(root, "model.safetensors.index.json",
                              pins["model.safetensors.index.json"], 1 << 20))
    cfg = json.loads(pinned(root, "config.json", pins["config.json"], 1 << 20))
    require(index["weight_map"][SOURCE] == SHARD and cfg["model_type"] ==
            cfg["text_config"]["model_type"] == "hy_v4_flash", "Not the pinned Hy4 gate source")
    with helpers.member(root, SHARD) as (f, size):
        fd = f.fileno()
        n, = struct.unpack("<Q", read(fd, 8, 0))
        require(0 < n <= helpers.HEADER_MAX and n + 8 <= size, "Invalid gate shard header")
        # Same strict bounded Halo parser as metadata planning; no buffered read.
        common.open = lambda path, mode: helpers.HeaderReader(f, n + 8)
        try:
            gate = common.load_safetensors_header(f"/proc/self/fd/{fd}")[SOURCE]
        finally:
            del common.open
        require(gate["dtype"] == "BF16" and gate["shape"] == shape[::-1] and
                gate["nbytes"] == 2 * common.product(shape), "Source gate shape/dtype mismatch")
        return read(fd, gate["nbytes"], gate["offset"])


def copy_span(src, dst, start, length, shift, block):
    lo = (start + block - 1) // block * block
    hi = (start + length) // block * block
    cloned = 0
    if lo < hi and shift % block == 0:
        try:
            fcntl.ioctl(dst, 0x4020940d, struct.pack("=qQQQ", src, lo, hi - lo, lo + shift))
            cloned = hi - lo
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTTY):
                raise
    ranges = [(start, lo - start), (hi, start + length - hi)] if cloned else [(start, length)]
    for a, size in ranges:
        for p, n in chunks(a, size):
            write(dst, read(src, n, p), p + shift)
    return cloned


def encode(quant, raw, shape, tile=128):
    np = quant.np
    width, rows = shape
    for row in range(0, rows, tile):
        block = raw[2 * width * row:2 * width * (row + tile)]
        values = (np.frombuffer(block, "<u2").astype(np.uint32) << 16).view(np.float32)
        require(np.isfinite(values).all(), "Nonfinite source gate")
        yield quant.encode(values.reshape(-1, width), 8)


def convert(control, preflight, library, source, common, pin, pins):
    with helpers.directory(preflight.parent) as root:
        report = json.loads(pinned(root, preflight.name, pins["preflight"], 1 << 20))
    tensors = report["tensors"]
    i, = (i for i, t in enumerate(tensors) if t["name"] == TARGET)
    target = tensors[i]
    shape = target["shape"]
    release = pins is PINS
    require(report["architecture"] == "hy_v4_flash" and report["source_revision"] == pin.REVISION
            and target["source"] == SOURCE and target["qtype"] == 30 and len(shape) == 2
            and 0 < target["nbytes"] <= 50331648, "Unexpected pinned gate descriptor")
    if release:
        require(len(tensors) == 1477 and shape == [3072, 8192], "Unexpected release geometry")
    data, disk = report["data_offset"], report["engram_file_offset"]
    require(0 < data <= 16 << 20, "Hy4 header exceeds its bound")
    at, old_bytes = data + target["offset"], target["nbytes"]
    new_bytes = common.qtype_nbytes(8, shape)
    shift = new_bytes - old_bytes
    end = at + old_bytes
    size = report["file_bytes"]
    require(shift < 0 and shift % 65536 == 0 and size + shift <= MAX_FILE, "Unexpected gate size shift")
    if release:
        require(size + shift == MAX_FILE, "Unexpected release output extent")
    with helpers.directory(control.parent) as control_root, helpers.member(control_root, control.name) as (f, length):
        src = f.fileno()
        require(size == length, "Wrong immutable control extent")
        header = read(src, data, 0)
        slots, fields, old_tables = layout(header, common, tensors, disk, size)
        if release:
            require(len(old_tables) == 4, "Unexpected release Engram inventory")
        with helpers.directory(source) as root:
            bf16 = source_gate(root, common, pins, shape)
        require(read(src, old_bytes, at) == bf16, "Source BF16 gate does not match control")
        # Retain a descriptor-bound read-only codec fd so the hash and dlopen
        # refer to the same file. The deployment also mounts it read-only.
        with helpers.directory(library.parent) as root, helpers.member(root, library.name) as (lib, n):
            require(n <= 1 << 20 and helpers.sha(read(lib.fileno(), n, 0)) == pins["quantizer"],
                    "Pinned quantizer SHA-256 mismatch")
            quant = common.Quantizer(f"/proc/self/fd/{lib.fileno()}")
            q8 = b"".join(encode(quant, bf16, shape))
            require(len(q8) == new_bytes, "Wrong encoded gate extent")
            candidate = bytearray(header)
            _, a, b = slots["halo.hy4.quantization"]
            require(header[a+8:b] == OLD, "Unexpected control quantization description")
            candidate[a+8:b] = NEW
            struct.pack_into("<Q", candidate, slots["halo.hy4.engram.file_offset"][1], disk + shift)
            updated = []
            for j, t in enumerate(tensors):
                new = dict(t)
                if j == i:
                    new.update(qtype=8, nbytes=new_bytes)
                if j > i:
                    new["offset"] += shift
                struct.pack_into("<IQ", candidate, fields[j], new["qtype"], new["offset"])
                updated.append(new)
            return write_verified(src, header, bytes(candidate), data, at, end, size,
                                  shift, bf16, q8, quant, shape, common, updated, old_tables, pins, pin)


def write_verified(src, header, candidate, data, at, end, size, shift, bf16, q8,
                   quant, shape, common, updated, old_tables, pins, pin):
    partial = Path(".gate.partial")
    fs = os.statvfs(".")
    require(fs.f_bavail * fs.f_frsize >= size + shift + (1 << 30), "Insufficient Hy4 scratch space")
    with partial.open("x+b", buffering=0) as f:
        dst = f.fileno()
        os.ftruncate(dst, size + shift)
        write(dst, candidate, 0)
        write(dst, q8, at)
        clones = copy_span(src, dst, data, at - data, 0, fs.f_frsize)
        clones += copy_span(src, dst, end, size - end, shift, fs.f_frsize)
        os.fsync(dst)
        actual = read(dst, data, 0)
        require(actual == candidate and os.fstat(dst).st_size == size + shift, "Bad Hy4 output header/size")
        _, _, tables = layout(actual, common, updated, old_tables[0]["file_offset"] + shift, size + shift)
        hc, hn = hashlib.sha256(header), hashlib.sha256(actual)
        prefix, suffix = hashlib.sha256(), hashlib.sha256()

        def same(start, length, delta, *hashes):
            for pos, count in chunks(start, length):
                b = read(src, count, pos)
                require(b == read(dst, count, pos + delta), f"Unchanged bytes differ at {pos}")
                for digest in (*hashes, hc, hn):
                    digest.update(b)

        same(data, at - data, 0, prefix)
        hc.update(bf16)
        pos = at
        for block in encode(quant, bf16, shape, tile=73):
            b = read(dst, len(block), pos)
            require(b == block, "Readback does not match source-derived Q8_0")
            hn.update(b)
            pos += len(block)
        require(pos == end + shift, "Wrong target end")
        same(end, old_tables[0]["file_offset"] - end, shift, suffix)
        for old, new in zip(old_tables, tables, strict=True):
            h = hashlib.sha256()
            same(old["file_offset"], old["bytes"], shift, suffix, h)
            new.update(control_file_offset=old["file_offset"], sha256=h.hexdigest())
        require(hc.hexdigest() == pins["control"], "Immutable control SHA-256 mismatch")
        os.fchmod(dst, 0o444)
        os.fsync(dst)
    return dict(
        kind="hy4_gate_encode", verified=True, runtime_ready=False, source_revision=pin.REVISION,
        control_sha256=hc.hexdigest(), preflight_sha256=pins["preflight"],
        quantizer_sha256=pins["quantizer"], source_metadata_sha256={
            k: pins[k] for k in ("config.json", "model.safetensors.index.json")},
        source_bf16_matches_control=True, unchanged_tensor_payloads=len(updated)-1,
        verified_tensor_spans=len(updated), data_offset=data, header_sha256=helpers.sha(candidate),
        resident_bytes=tables[0]["file_offset"], resident_delta_bytes=shift,
        base_resident_bytes=old_tables[0]["file_offset"], reflink_bytes=clones,
        unchanged_prefix=dict(control_offset=data, file_offset=data, bytes=at-data, sha256=prefix.hexdigest()),
        unchanged_suffix=dict(control_offset=end, file_offset=end+shift, bytes=size-end, sha256=suffix.hexdigest()),
        engram=dict(file_offset=tables[0]["file_offset"], disk_bytes=size-old_tables[0]["file_offset"], tables=tables),
        target=dict(name=TARGET, source=SOURCE, source_shard=SHARD, shape=shape, file_offset=at, qtype_before=30,
                    qtype=8, bytes=len(q8), source_bf16_sha256=helpers.sha(bf16), sha256=helpers.sha(q8)),
        artifact=dict(path=str(Path.cwd()/"hy4-gate-q8.gguf"), kind="gguf", bytes=size+shift, sha256=hn.hexdigest()),
        receipt_path=str(Path.cwd()/"hy4-gate-q8.json"),
    )


def limit(which, n):
    _, hard = resource.getrlimit(which)
    value = min(hard, n) if hard != resource.RLIM_INFINITY else n
    resource.setrlimit(which, (value, value))


def main(*, pins=PINS):
    limit(resource.RLIMIT_AS, 1 << 30)
    limit(resource.RLIMIT_FSIZE, MAX_FILE)
    signal.alarm(1800)
    host, source, tools, control, preflight, library = sys.argv[1:]
    raw = sys.stdin.buffer.read(65537)
    require(len(raw) <= 65536, "Hy4 job envelope too large")
    job = json.loads(raw)
    inputs = job.get("inputs", [])
    require(job.get("kind") == "convert" and job.get("prompt") is None
            and job.get("options") == {"hy4_overrides": {TARGET: "Q8_0"}}
            and len(inputs) == 1 and inputs[0].get("host") == host
            and inputs[0].get("path") == source and inputs[0].get("kind") == "hf"
            and inputs[0].get("integrity_basis") == "hy4_metadata_only",
            "Invalid dedicated Hy4 encode request")
    _, pin = helpers.halo_modules(Path(tools))
    common = sys.modules["gguf_common"]
    result = convert(Path(control), Path(preflight), Path(library), Path(source), common, pin, pins)
    payload = json.dumps(result, indent=2).encode() + b"\n"
    require(len(payload) <= 1 << 20, "Hy4 receipt exceeds its bound")
    with open("hy4-gate-q8.json", "xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.rename(".gate.partial", "hy4-gate-q8.gguf")
    helpers_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(helpers_fd)
    finally:
        os.close(helpers_fd)
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
