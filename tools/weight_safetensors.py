"""Bounded SafeTensors header/range access and verified streaming publication."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

INDEX = "model.safetensors.index.json"
HEADER = 4 << 20
METADATA = 16 << 20
MANIFEST = 64 << 20
UNPROVEN_BYTES = 64 << 20
MAX_BYTES = 2 << 40
MAX_MEMBERS = 16384
MAX_TENSORS = 100000
CHUNK = 1 << 20
OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
OPEN_FILE = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
WIDTHS = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}


def need(test, message):
    if not test:
        raise ValueError(message)


def pathsafe(name):
    return (isinstance(name, str) and len(name.encode("utf-8")) <= 4096
            and all(c.isprintable() and c != "\\" for c in name)
            and all(p not in ("", ".", "..") for p in name.split("/")))


def tensor_name(name):
    return 0 < len(name) <= 95 and name.isascii() and name.isprintable() and name != "__metadata__"


def parse_json(raw):
    def unique(pairs):
        obj = dict(pairs)
        need(len(obj) == len(pairs), "duplicate SafeTensors metadata key")
        return obj

    def nonfinite(value):
        raise ValueError("non-finite JSON literal in SafeTensors metadata")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def stamp(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def read_at(fd, offset, count):
    raw = os.pread(fd, count, offset)
    need(len(raw) == count, "truncated SafeTensors file")
    return raw


def hash_fd(fd, size):
    digest = hashlib.sha256()
    for start in range(0, size, CHUNK):
        digest.update(read_at(fd, start, min(CHUNK, size - start)))
    return digest.hexdigest()


def check_verity(fd):
    # Linux FS_IOC_MEASURE_VERITY: just confirm the inode is immutable.
    # The operator manifest (bound to this inode by the worker) supplies the
    # ordinary SHA-256, which is distinct from the measured verity root.
    output = bytearray(struct.pack("=HH32s", 0, 32, b""))
    fcntl.ioctl(fd, 0xC0046686, output, True)
    need(struct.unpack_from("=HH", output) == (1, 32),
         "SafeTensors requires immutable SHA-256 fs-verity proof")


def tree_digest(rows):
    digest = hashlib.sha256()
    for path, size, sha in rows:
        digest.update(f"{path}\0{size}\0{sha}\n".encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def open_at(root, parts, flags):
    """Walk components with O_NOFOLLOW; parent descriptors anchor lookup."""
    fd = os.dup(root)
    try:
        for i, part in enumerate(parts):
            child = os.open(part, flags if i == len(parts) - 1 else OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_path(path, flags):
    with root_fd() as root:
        parts = Path(os.path.abspath(path)).parts[1:]
        need(".." not in Path(path).parts, "parent traversal in SafeTensors path")
        with open_at(root, parts, flags) as fd:
            yield fd


@contextmanager
def root_fd():
    fd = os.open("/", OPEN_DIR)
    try:
        yield fd
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Tensor:
    name: str
    member: str
    dtype: str
    shape: tuple
    start: int


def parse_header(read, member, size, budget):
    length, = struct.unpack("<Q", read(0, 8))
    need(2 <= length <= min(HEADER, budget, size - 8), "SafeTensors header exceeds bounds")
    raw = read(8, length)
    need(raw.startswith(b"{") and raw.endswith((b"}", b" ")), "invalid SafeTensors header")
    header = parse_json(raw)
    need(isinstance(header, dict), "invalid SafeTensors header object")
    meta = header.pop("__metadata__", {})
    need(isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values()),
         "invalid SafeTensors __metadata__")
    need(0 < len(header) <= MAX_TENSORS, "invalid SafeTensors tensor count")
    result, ranges = {}, []
    for name, entry in header.items():
        need(tensor_name(name), "invalid SafeTensors tensor name")
        need(isinstance(entry, dict) and set(entry) == {"dtype", "shape", "data_offsets"},
             "invalid SafeTensors tensor fields")
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        need(isinstance(dtype, str) and dtype in WIDTHS, "unsupported SafeTensors dtype layout")
        need(isinstance(shape, list) and len(shape) <= 8 and
             all(type(n) is int and 0 <= n <= MAX_BYTES for n in shape), "invalid SafeTensors shape")
        need(isinstance(offsets, list) and len(offsets) == 2 and
             all(type(n) is int and 0 <= n <= size - 8 - length for n in offsets),
             "invalid SafeTensors offsets")
        start, end = offsets
        need(end - start == math.prod(shape) * WIDTHS[dtype], "SafeTensors payload size mismatch")
        ranges.append((start, end))
        result[name] = Tensor(name, member, dtype, tuple(shape), 8 + length + start)
    end = 0
    for start, next_end in sorted(ranges):
        need(start == end, "SafeTensors payload has a gap or overlap")
        end = next_end
    need(8 + length + end == size, "SafeTensors file has unindexed bytes")
    return result, length, end


def read_manifest(path):
    with open_path(path, OPEN_FILE) as fd:
        meta = os.fstat(fd)
        need(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1 and meta.st_uid == os.geteuid()
             and meta.st_mode & 0o077 == 0 and 0 < meta.st_size <= MANIFEST,
             "expected a private source manifest file")
        raw = read_at(fd, 0, meta.st_size)
        need(stamp(meta) == stamp(os.fstat(fd)), "source manifest changed during read")
    rows = parse_json(raw)
    need(isinstance(rows, list) and 0 < len(rows) <= MAX_MEMBERS, "invalid source manifest")
    entries = {}
    previous = ""
    for row in rows:
        fields = {"path", "bytes", "sha256"}
        need(isinstance(row, dict) and set(row) in (
            fields, fields | {"device", "inode"}, fields | {"chunk_sha256"}),
            "invalid source manifest entry")
        name, length, sha = row["path"], row["bytes"], row["sha256"]
        need(pathsafe(name) and name > previous and type(length) is int and 0 <= length <= MAX_BYTES
             and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha),
             "invalid or unsorted source manifest entry")
        if "inode" in row:
            need(all(type(row[k]) is int and row[k] >= 0 for k in ("device", "inode"))
                 and row["inode"] > 0, "invalid source inode proof")
        if "chunk_sha256" in row:
            chunks = row["chunk_sha256"]
            need(isinstance(chunks, list) and len(chunks) == (length + CHUNK - 1) // CHUNK
                 and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in chunks),
                 "invalid source chunk SHA-256 list")
        entries[name] = row
        previous = name
    return entries


class Files:
    """Metadata stays resident, payload reads use checked, bounded file descriptors."""

    def __init__(self, path, expected, manifest=None):
        self.path = Path(path)
        meta = self.path.lstat()
        self.directory = stat.S_ISDIR(meta.st_mode)
        self.parent = self.path if self.directory else self.path.parent
        self.stat = {}
        self.dirs = {}
        if self.directory:
            with open_path(self.parent, OPEN_DIR) as fd:
                self._walk(fd)
            need(INDEX in self.stat, "missing indexed HF SafeTensors manifest")
        else:
            need(self.path.suffix == ".safetensors" and stat.S_ISREG(meta.st_mode)
                 and meta.st_nlink == 1, "expected unlinked standalone .safetensors file")
            self.stat[self.path.name] = meta
        self.names = sorted(self.stat)
        total = sum(self.stat[n].st_size for n in self.names)
        need(0 < total <= MAX_BYTES, "SafeTensors file set exceeds size limit")
        if manifest is not None:
            entries = read_manifest(manifest)
            if not self.directory:
                need(len(entries) == 1, "standalone SafeTensors needs one manifest entry")
                entries = {self.path.name: next(iter(entries.values()))}
            need(entries.keys() == self.stat.keys(), "source manifest membership differs")
            for name, entry in entries.items():
                st = self.stat[name]
                need(st.st_size == entry["bytes"] and
                     ("inode" not in entry or (st.st_dev, st.st_ino) == (entry["device"], entry["inode"])),
                     "SafeTensors source size or inode differs from proof")
        else:
            entries = {}
        # Chunk hashes are trusted only for manifests issued by the worker
        # from an operator-pinned index bound to the full source identity.
        self.chunks = {name: entry["chunk_sha256"] for name, entry in entries.items()
                       if "chunk_sha256" in entry}
        self._verified_chunk = None
        unproven = [name for name in self.names
                    if name not in self.chunks and "inode" not in entries.get(name, {})]
        need(sum(self.stat[n].st_size for n in unproven) <= UNPROVEN_BYTES,
              "large SafeTensors source requires operator-verified immutable inode manifest "
              "or authenticated chunk manifest")
        self.hashes = {}
        for name in self.names:
            entry = entries.get(name)
            if name in self.chunks:
                self.hashes[name] = entry["sha256"]
            elif entry is not None and "inode" in entry:
                with self.open_member(name) as fd:
                    try:
                        check_verity(fd)
                    except OSError as exc:
                        raise ValueError("SafeTensors requires immutable fs-verity proof") from exc
                self.hashes[name] = entry["sha256"]
            else:
                with self.open_member(name) as fd:
                    actual = hash_fd(fd, self.stat[name].st_size)
                need(entry is None or actual == entry["sha256"], "SafeTensors source SHA-256 mismatch")
                self.hashes[name] = actual
        computed = self.digest(self.hashes)
        need(expected == computed, "SafeTensors source SHA-256 mismatch")
        self.tensors = self._headers()
        self.check_current()

    def _walk(self, fd, prefix=""):
        self.dirs[prefix] = os.fstat(fd)
        with os.scandir(fd) as children:
            for entry in children:
                name = f"{prefix}/{entry.name}" if prefix else entry.name
                need(pathsafe(name) and name.count("/") < 32, "unsafe indexed HF member name")
                st = entry.stat(follow_symlinks=False)
                need(len(self.stat) + len(self.dirs) < MAX_MEMBERS, "too many indexed HF members")
                if stat.S_ISDIR(st.st_mode):
                    with open_at(fd, [entry.name], OPEN_DIR) as child:
                        need(stamp(st) == stamp(os.fstat(child)), "indexed HF directory changed")
                        self._walk(child, name)
                else:
                    need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
                         "indexed HF requires regular files without links")
                    self.stat[name] = st

    @contextmanager
    def open_member(self, name):
        with open_path(self.parent, OPEN_DIR) as root:
            if self.directory:
                need(stamp(os.fstat(root)) == stamp(self.dirs[""]), "indexed HF directory changed")
            with open_at(root, name.split("/"), OPEN_FILE) as fd:
                expected = stamp(self.stat[name])
                current = os.fstat(fd)
                need(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                     and stamp(current) == expected, "SafeTensors input changed before read")
                yield fd
                need(stamp(os.fstat(fd)) == expected, "SafeTensors input changed during read")

    def check_current(self):
        with open_path(self.parent, OPEN_DIR) as root:
            for name, expected in self.dirs.items():
                with open_at(root, name.split("/") if name else [], OPEN_DIR) as fd:
                    need(stamp(os.fstat(fd)) == stamp(expected), "indexed HF directory changed")
        for name in self.names:
            with self.open_member(name):
                pass

    def digest(self, hashes):
        if not self.directory:
            return hashes[self.names[0]]
        return tree_digest((name, self.stat[name].st_size, hashes[name]) for name in self.names)

    def _headers(self):
        if self.directory:
            size = self.stat[INDEX].st_size
            need(0 < size <= HEADER, "indexed HF weight map exceeds bounds")
            index = parse_json(self.range(INDEX, 0, size))
            mapping = index.get("weight_map") if isinstance(index, dict) else None
            need(isinstance(mapping, dict) and 0 < len(mapping) <= MAX_TENSORS,
                 "invalid indexed HF weight map")
            for name, member in mapping.items():
                need(tensor_name(name) and pathsafe(member) and member.endswith(".safetensors")
                     and "/" not in member and member in self.stat, "unsafe indexed shard or tensor")
            shards = sorted(set(mapping.values()))
        else:
            mapping = None
            shards = self.names
        tensors, total, budget = {}, 0, METADATA - (size if mapping is not None else 0)
        for name in shards:
            with self.open_member(name) as fd:
                def read(offset, count):
                    return self._read_member(fd, name, offset, count)

                data, length, payload = parse_header(read, name, self.stat[name].st_size, budget)
            need(len(tensors) + len(data) <= MAX_TENSORS, "SafeTensors tensor count exceeds bounds")
            need(not tensors.keys() & data.keys(), "duplicate indexed SafeTensors tensor")
            tensors.update(data)
            total += payload
            budget -= length
        if mapping is not None:
            need(mapping == {name: t.member for name, t in tensors.items()},
                 "weight map does not match SafeTensors shard headers")
            meta = index.get("metadata", {})
            need(isinstance(meta, dict), "invalid index metadata")
            if "total_size" in meta:
                count = meta["total_size"]
                # HF indices may spell an integral byte count with a fractional
                # JSON representation. The 2 TiB limit is inside the exact
                # integer range of binary64.
                need(type(count) in (int, float) and count == total,
                     "index metadata total_size differs")
        return tensors

    def range(self, member, offset, count):
        with self.open_member(member) as fd:
            return self._read_member(fd, member, offset, count)

    def _read_member(self, fd, member, offset, count):
        size = self.stat[member].st_size
        need(0 <= offset <= size and 0 <= count <= size - offset, "invalid SafeTensors read range")
        if member not in self.chunks or count == 0:
            return read_at(fd, offset, count)
        end = offset + count
        pieces = []
        for block in range(offset // CHUNK, (end - 1) // CHUNK + 1):
            start = block * CHUNK
            key = (member, block)
            cached = self._verified_chunk
            if cached is not None and cached[0] == key:
                raw = cached[1]
            else:
                raw = read_at(fd, start, min(CHUNK, size - start))
                need(hashlib.sha256(raw).hexdigest() == self.chunks[member][block],
                     f"SafeTensors chunk SHA-256 mismatch: {member} chunk {block}")
                # Keep only the last verified immutable buffer. Headers and
                # adjacent tile cells can reuse it without another disk read.
                self._verified_chunk = (key, raw)
            pieces.append(raw[max(0, offset - start):end - start])
        return b"".join(pieces)

    def _write_member(self, name, dest, patches):
        digest, changed = hashlib.sha256(), hashlib.sha256()
        size = self.stat[name].st_size
        pieces = sorted(patches.items())
        previous = 0
        def create(path, flags):
            return os.open(path, flags | os.O_NOFOLLOW, 0o600)

        with self.open_member(name) as fd, open(dest, "xb", opener=create) as out:
            for start, replacement in (*pieces, (size, b"")):
                need(previous <= start and start + len(replacement) <= size, "invalid edit range")
                for offset in range(previous, start, CHUNK):
                    data = read_at(fd, offset, min(CHUNK, start - offset))
                    digest.update(data)
                    changed.update(data)
                    out.write(data)
                if replacement:
                    digest.update(read_at(fd, start, len(replacement)))
                    changed.update(replacement)
                    out.write(replacement)
                previous = start + len(replacement)
            need(digest.hexdigest() == self.hashes[name], "source changed since verified manifest")
            out.flush()
            os.fsync(out.fileno())
        with open_path(dest, OPEN_FILE) as fd:
            need(os.fstat(fd).st_size == size and hash_fd(fd, size) == changed.hexdigest(),
                 "SafeTensors output differs from verified copy")
        return changed.hexdigest()

    def publish(self, dest, patches):
        dest = Path(dest)
        need(not os.path.lexists(dest), "destination already exists")
        with open_path(dest.parent, OPEN_DIR) as parent:
            if self.directory:
                need(not Path(os.path.abspath(dest)).is_relative_to(Path(os.path.abspath(self.path))),
                     "destination cannot be inside the source directory")
            space = os.fstatvfs(parent)
            need(space.f_bavail * space.f_frsize >= sum(st.st_size for st in self.stat.values()),
                 "insufficient disk for complete SafeTensors copy")
        with tempfile.TemporaryDirectory(prefix=".weight-safetensors-", dir=dest.parent) as scratch:
            work = Path(scratch) / "result"
            if self.directory:
                work.mkdir(mode=0o700)
                for name in sorted(self.dirs):
                    if name:
                        (work / name).mkdir(mode=0o700)
            by_file = {}
            for (name, offset), replacement in patches.items():
                by_file.setdefault(name, {})[offset] = replacement
            hashes = {}
            for name in self.names:
                target = work / name if self.directory else work
                hashes[name] = self._write_member(name, target, by_file.get(name, {}))
            self.check_current()
            if self.directory:
                for name in sorted(self.dirs, reverse=True):
                    with open_path(work / name, OPEN_DIR) as fd:
                        os.fsync(fd)
            rename_noreplace(work, dest)
        return self.digest(hashes)


def rename_noreplace(source, dest):
    # Do not fall back to os.rename (which can clobber an empty destination).
    call = ctypes.CDLL(None, use_errno=True).renameat2
    call.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    with open_path(source.parent, OPEN_DIR) as sfd, open_path(dest.parent, OPEN_DIR) as dfd:
        if call(sfd, os.fsencode(source.name), dfd, os.fsencode(dest.name), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), dest)
        os.fsync(dfd)
