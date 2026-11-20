"""Worker-only metadata planner for the pinned Hy4 GGUF v1 layout."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import struct
import sys
import types

TARGET = "blk.1.ffn_down_exps.weight"
OVERRIDE = {TARGET: "Q8_0"}
OLD = b"Q4_K routed experts only; other tensors raw"
NEW = b"Q8_0 block1 down; Q4_K gate/up; other raw  "
PINS = {
    "gguf_common": "674b07d8bcd136528c1ae32464ee755c6f11c511fa14094fd33d76bd610afe5f",
    "hy4_schema": "e0796364c331006d1f05b7ae8a0b831796203c9ead53f25da04a0193c6cc3af4",
    "hy4_quantize": "4f03b9132f4111eab84275ca4683194ffbe954a238f5ce9e11fdcf878b64039f",
}
HEADER_MAX = 1 << 20
METADATA_MAX = 16 << 20


def require(ok, error):
    if not ok:
        raise ValueError(error)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def member(root_fd, name):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", ".."),
            "Invalid Hy4 metadata filename")
    fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=root_fd)
    with os.fdopen(fd, "rb", buffering=0) as f:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "Hy4 input is not a regular file")
        yield f, before.st_size
        after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after), "Hy4 metadata changed during planning")


@contextmanager
def directory(path):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Expected a canonical directory")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield fd
        current, opened = path.stat(), os.fstat(fd)
        require((current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                "Hy4 directory changed during planning")
    finally:
        os.close(fd)


def halo_modules(path):
    # Validate the exact bytes before executing them. Do not import from a
    # mutable search path, from the source snapshot, or from existing pyc files.
    with directory(path) as root:
        for name, digest in PINS.items():
            filename = name + ".py"
            with member(root, filename) as (f, size):
                require(size <= HEADER_MAX, "Hy4 planner module is too large")
                data = f.read(HEADER_MAX + 1)
                require(sha(data) == digest, f"{filename} does not match the pinned planner")
            mod = types.ModuleType(name)
            mod.__file__ = str(path / filename)
            sys.modules[name] = mod
            exec(compile(data, mod.__file__, "exec"), mod.__dict__)
    return sys.modules["hy4_quantize"], sys.modules["hy4_schema"]


class HeaderReader:
    """Constrain Halo's parser to exact, unbuffered reads of header bytes."""

    def __init__(self, file, length):
        self.file = file
        self.remaining = length

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, n):
        require(0 <= n <= self.remaining, "SafeTensors header exceeds its approved extent")
        self.remaining -= n
        return self.file.read(n)


class Inventory:
    def __init__(self, root, index, common, max_shards):
        weight_map = json.loads(index)["weight_map"]
        require(isinstance(weight_map, dict) and 0 < len(weight_map) <= 4096,
                "Invalid Hy4 index inventory size")
        shards = sorted(set(weight_map.values()))
        require(0 < len(shards) <= max_shards and all(
            re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", s) for s in shards
        ), "Invalid Hy4 shard inventory")
        self.tensors = {}
        self.header_bytes = 0
        for shard in shards:
            with member(root, shard) as (f, size):
                raw = os.pread(f.fileno(), 8, 0)
                require(len(raw) == 8, "Short SafeTensors header")
                n, = struct.unpack("<Q", raw)
                self.header_bytes += n + 8
                require(0 < n <= HEADER_MAX and n + 8 <= size and self.header_bytes <= METADATA_MAX,
                        "Hy4 SafeTensors metadata exceeds the read budget")
                # getsize() and reads refer to the already-open regular file.
                # The reader runs no buffered read that could consume payload.
                fd_name = f"/proc/self/fd/{f.fileno()}"
                prior = common.__dict__.get("open")
                try:
                    common.open = lambda path, mode: HeaderReader(f, n + 8)
                    tensors = common.load_safetensors_header(fd_name)
                finally:
                    if prior is None:
                        del common.open
                    else:
                        common.open = prior
            for name, info in tensors.items():
                require(weight_map.get(name) == shard and name not in self.tensors,
                        "Hy4 shard differs from pinned index")
                self.tensors[name] = dict(info, shard=shard)
        require(self.tensors.keys() == weight_map.keys(), "Hy4 index inventory is incomplete")

    def info(self, name):
        return self.tensors[name]


def plan(source, q, pin, override, *, pins=None, schema=None):
    require(override == OVERRIDE, "Expected the exact block-1 down Q8_0 override")
    common = sys.modules["gguf_common"]
    with directory(source) as root:
        metadata = {}
        total = 0
        for name, digest in (pin.SOURCE_FILES if pins is None else pins).items():
            with member(root, name) as (f, size):
                total += size
                require(total <= METADATA_MAX, "Hy4 source metadata exceeds the read budget")
                data = f.read(size + 1)
                require(len(data) == size and sha(data) == digest, f"{name} does not match the pinned release")
                metadata[name] = data
        index = metadata.pop("model.safetensors.index.json")
        config = json.loads(metadata["config.json"])
        require(config["model_type"] == config["text_config"]["model_type"] == pin.ARCHITECTURE,
                "Not a pinned Hy4 configuration")
        db = Inventory(root, index, common, 148)
    schema = (schema or pin.release_schema)(db)
    base, excluded = q.build_plan(db, schema)
    old_header, old_disk = q.build_header(pin.ARCHITECTURE, metadata, base)
    tensors, _ = q.build_plan(db, schema)
    target, = (t for t in tensors if t.name == TARGET)
    original = next(t for t in base if t.name == TARGET)
    require(target.source == "model.layers.1.mlp.experts.down_proj"
            and target.role == "experts" and target.qtype == 12 and target.row_start == 0
            and db.info(target.source)["dtype"] == "BF16", "Not the pinned BF16 down projection")
    target.qtype = common.QTYPE_Q8_0
    target.nbytes = common.qtype_nbytes(target.qtype, target.shape)
    offset = 0
    for item in tensors:
        item.offset = offset
        offset += common.align(item.nbytes)
    header, disk = q.build_header(pin.ARCHITECTURE, metadata, tensors)
    require(len(header) == len(old_header) and len(NEW) == len(OLD) and header.count(OLD) == 1,
            "Hy4 quantization description would change the header size")
    header = header.replace(OLD, NEW)
    disk_items = [t for t in tensors if t.role == "engram_disk"]
    end = disk
    tables = []
    for item in disk_items:
        start = len(header) + item.offset
        require(item.qtype == 30 and item.shape[0] == 96 and start == end,
                "Invalid planned BF16 Engram extent")
        end += item.nbytes
        tables.append(dict(name=item.name, source=item.source, qtype=item.qtype, rows=item.shape[1],
                           file_offset=start, bytes=item.nbytes))
    require(disk % 65536 == 0 and end == q.tensor_end(header, tensors[-1]), "Invalid Engram alignment or tail")
    if pins is None:
        require(len(tables) == 4 and len(tensors) == 1477, "Invalid pinned Hy4 descriptor inventory")
    return dict(
        kind="hy4_plan", format="halo_hy4_gguf_v1", architecture=pin.ARCHITECTURE,
        source_revision=pin.REVISION, source_inventory_sha256=pin.INVENTORY_SHA256 if pins is None else None,
        source_metadata_sha256=pin.SOURCE_FILES if pins is None else pins,
        planner_sha256=PINS, payload_verified=False, runtime_ready=False, text_only=True,
        source_metadata_read_bytes=total + db.header_bytes + 8 * len({v["shard"] for v in db.tensors.values()}),
        tensor_count=len(tensors), excluded_counts={role: len(items) for role, items in excluded.items()},
        data_offset=len(header), header_sha256=sha(header), quantization=NEW.decode("ascii"), file_bytes=end,
        base_file_bytes=q.tensor_end(old_header, base[-1]), base_resident_bytes=old_disk,
        resident_bytes=disk, resident_delta_bytes=disk - old_disk,
        override={**asdict(target), "shape": list(target.shape), "qtype_before": original.qtype,
                  "bytes_before": original.nbytes, "bytes": target.nbytes,
                  "file_offset": len(header) + target.offset},
        engram=dict(file_offset=disk, disk_bytes=end - disk, row_bytes=192, tables=tables),
    )


def main():
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    signal.alarm(20)
    host, source, tools = sys.argv[1:]
    raw = sys.stdin.buffer.read(65537)
    require(len(raw) <= 65536, "Hy4 worker job envelope too large")
    job = json.loads(raw)
    options = job.get("options")
    inputs = job.get("inputs")
    require(job.get("kind") == "plan" and job.get("prompt") is None
            and options == {"hy4_overrides": OVERRIDE}
            and isinstance(inputs, list) and len(inputs) == 1
            and inputs[0].get("kind") == "hf" and inputs[0].get("host") == host
            and inputs[0].get("path") == source, "Invalid pinned Hy4 plan request")
    q, pin = halo_modules(Path(tools))
    print(json.dumps(plan(Path(source), q, pin, options["hy4_overrides"]), separators=(",", ":")))


if __name__ == "__main__":
    main()
