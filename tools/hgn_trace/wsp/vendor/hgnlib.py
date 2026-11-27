"""Minimal HGN container tools following jtsylve/hgn-spec 1.0.1 CONTAINER.md.

patch(src, dst, replacements, drop=()) copies src, appends replacement payloads
at 64-byte boundaries, and updates the tensor table and header atomically.
"""
import os, shutil, struct, subprocess, tempfile
import numpy as np

HDR, ENT = 0x68, 0xA0


class Entry:
    __slots__ = ("name", "store", "shape", "offset", "size", "checksum", "variant")

    def pack(self):
        b = bytearray(ENT)
        n = self.name.encode()
        assert 1 <= len(n) <= 95
        b[:len(n)] = n
        struct.pack_into("<II", b, 0x60, self.store, len(self.shape))
        struct.pack_into("<4q", b, 0x68, *(list(self.shape) + [0] * (4 - len(self.shape))))
        struct.pack_into("<QQII", b, 0x88, self.offset, self.size, self.checksum, self.variant)
        return bytes(b)


def read(path):
    with open(path, "rb") as f:
        h = f.read(HDR)
        if len(h) != HDR:
            raise ValueError(f"truncated HGN header: {path}")
        magic, ver, n, toff, doff, fsize = struct.unpack("<IIQQQQ", h[:0x28])
        actual_size = os.path.getsize(path)
        if magic != 0x314E4748 or fsize != actual_size:
            raise ValueError(f"invalid HGN magic or file size: {path}")
        ident = h[0x28:0x68].rstrip(b"\0").decode()
        f.seek(toff)
        ents = []
        for _ in range(n):
            b = f.read(ENT)
            if len(b) != ENT:
                raise ValueError(f"truncated HGN tensor table: {path}")
            e = Entry()
            e.name = b[:96].split(b"\0")[0].decode()
            e.store, rank = struct.unpack_from("<II", b, 0x60)
            if rank > 4:
                raise ValueError(f"invalid tensor rank for {e.name}: {rank}")
            e.shape = tuple(struct.unpack_from("<4q", b, 0x68)[:rank])
            e.offset, e.size, e.checksum, e.variant = struct.unpack_from("<QQII", b, 0x88)
            if e.offset + e.size > actual_size:
                raise ValueError(f"payload outside HGN file: {e.name}")
            ents.append(e)
    return ver, ident, doff, ents


def xor_fold(buf):
    a = np.frombuffer(buf, dtype=np.uint8)
    pad = (-len(a)) % 4
    if pad:
        a = np.concatenate([a, np.zeros(pad, np.uint8)])
    return int(np.bitwise_xor.reduce(a.view("<u4"))) if len(a) else 0


def payload(path, e):
    return np.memmap(path, dtype=np.uint8, mode="r", offset=e.offset, shape=(e.size,))


def supports_reflink(src, directory):
    """Probe whether cp can clone src into the output filesystem."""
    with tempfile.TemporaryDirectory(prefix=".hgn-reflink-check-", dir=directory) as probe_dir:
        target = os.path.join(probe_dir, "probe.hgn")
        try:
            subprocess.run(
                ["cp", "--reflink=always", "--", src, target],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return False
    return True


def output_allocation(src, directory, final_size, reflink_supported=None):
    """Estimate new disk allocation: appended data on CoW, full output otherwise."""
    if reflink_supported is None:
        reflink_supported = supports_reflink(src, directory)
    if reflink_supported:
        return max(0, final_size - os.path.getsize(src)), True
    return final_size, False

def patch(
    src, dst, replacements, drop=(), ident=None, log=print,
    estimated_size=None, margin_bytes=None, reflink_supported=None,
):
    """Append encoded payloads, preserving the input and atomically publishing dst.

    replacements maps HGN names to callables returning
    ``(store, variant, shape, bytes)``. Each callable runs one at a time.
    """
    if os.path.lexists(dst):
        raise FileExistsError(f"output already exists: {dst}")
    parent = os.path.dirname(os.path.abspath(dst))
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"output directory does not exist: {parent}")
    ver, old_ident, doff, ents = read(src)
    src_size = os.path.getsize(src)
    if estimated_size is None:
        estimated_size = src_size * 2
    output_bytes, reflink_supported = output_allocation(
        src, parent, estimated_size, reflink_supported
    )
    if margin_bytes is None:
        margin_bytes = max(1 << 20, output_bytes // 20)
    required = output_bytes + margin_bytes
    free = shutil.disk_usage(parent).free
    if free < required:
        raise OSError(
            f"insufficient free space for {dst}: need {required} bytes "
            f"({output_bytes} output allocation + {margin_bytes} margin), have {free}"
        )
    tmp = dst + ".tmp"
    if os.path.lexists(tmp):
        raise FileExistsError(f"temporary output already exists: {tmp}")
    subprocess.run(["cp", "--reflink=auto", "--", src, tmp], check=True)
    drop = set(drop)
    keep = [e for e in ents if e.name not in drop]
    byname = {e.name: e for e in keep}
    if not set(replacements).issubset(byname):
        raise ValueError(f"replacement names not in source HGN: {sorted(set(replacements) - set(byname))}")
    with open(tmp, "r+b") as f:
        end = os.path.getsize(tmp)
        for name, make in replacements.items():
            store, variant, shape, data = make()
            e = byname[name]
            if tuple(shape) != e.shape:
                raise ValueError(f"{name}: replacement shape {tuple(shape)} != HGN shape {e.shape}")
            end = (end + 63) // 64 * 64
            f.seek(end)
            f.write(data)
            e.store, e.variant = store, variant
            e.offset, e.size, e.checksum = end, len(data), xor_fold(data)
            end += len(data)
            log(f"  wrote {name}: store {store}/{variant}, {len(data)} bytes")
            del data
        end = (end + 63) // 64 * 64
        f.truncate(end)
        if HDR + ENT * len(keep) > doff:
            raise ValueError("updated tensor table overlaps HGN payload data")
        f.seek(HDR)
        f.write(b"".join(e.pack() for e in keep))
        f.write(b"\0" * (ENT * (len(ents) - len(keep))))
        idb = (ident or old_ident).encode()
        if len(idb) > 64:
            raise ValueError("HGN identity is longer than 64 bytes")
        f.seek(0)
        f.write(struct.pack("<IIQQQQ", 0x314E4748, ver, len(keep), HDR, doff, end) + idb + b"\0" * (64 - len(idb)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    dir_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

