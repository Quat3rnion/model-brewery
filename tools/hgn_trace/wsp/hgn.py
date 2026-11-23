"""Read access to a Halogen .hgn checkpoint as float32 numpy tensors.

Wraps the vendored Apache-2.0 hgn tooling (wsp/vendor). Supported stores:
Bf16(0), Q4Cp(5), Q8G64(7), Ht(16/0x1208), I4r(23), and the raw F16/F32/I64
planes. Q6G64(24) has no decoder yet (hyper-connection mixers only).
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np

_VENDOR = str(Path(__file__).with_name("vendor"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import hgnlib as H  # noqa: E402
import hgnenc as C  # noqa: E402
import hgnht as T  # noqa: E402

STORE_NAMES = {0: "Bf16", 5: "Q4Cp", 7: "Q8G64", 16: "Ht", 23: "I4r", 24: "Q6G64"}
DEFAULT = os.environ.get(
    "WSP_CHECKPOINT",
    "/home/quaternion/models/Signal-3.8-Flash-Next-HGN-v2-abliterated-4d30467d-20261014T203725Z/model.hgn",
)
PLAIN = "/home/quaternion/models/Signal-3.8-Flash-Next-HGN-v2-4d30467d-20261014T203725Z/model.hgn"


def _raw_plane(buf: bytes, shape, nbytes_per: int):
    n = int(np.prod(shape))
    if len(buf) == n * 2 and nbytes_per in (0, 2):
        return np.frombuffer(buf, "<f2").astype(np.float32).reshape(shape)
    if len(buf) == n * 4:
        return np.frombuffer(buf, "<f4").reshape(shape).copy()
    if len(buf) == n * 8:
        return np.frombuffer(buf, "<i8").reshape(shape).copy()
    raise ValueError(f"cannot interpret raw plane of {len(buf)} bytes as {shape}")


class Checkpoint:
    def __init__(self, path: str = DEFAULT):
        self.path = path
        self.version, self.ident, self.data_offset, entries = H.read(path)
        self.entries = {e.name: e for e in entries}

    # ------------------------------------------------------------ metadata
    def names(self, prefix: str = ""):
        return [n for n in self.entries if n.startswith(prefix)]

    def info(self, name: str) -> dict:
        e = self.entries[name]
        return {"name": name, "store": STORE_NAMES.get(e.store, e.store), "variant": e.variant,
                "shape": list(e.shape), "bytes": e.size, "offset": e.offset}

    def raw(self, name: str) -> bytes:
        return bytes(H.payload(self.path, self.entries[name]))

    # ------------------------------------------------------------ decoding
    def _side(self, root: str, suffix: str):
        return np.frombuffer(self.raw(root + suffix), "<f2").astype(np.float32)

    def get(self, name: str, lo: int | None = None, hi: int | None = None) -> np.ndarray:
        """Decode a tensor (or first-axis slice [lo:hi]) to float32."""
        e = self.entries[name]
        shape = e.shape
        if lo is None:
            lo, hi = 0, shape[0] if shape else 1
        if e.store == 0:
            k = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            p = H.payload(self.path, e)
            a = np.frombuffer(bytes(p[lo * k * 2: hi * k * 2]), "<u2").astype(np.uint32)
            return (a << 16).view(np.float32).reshape((hi - lo,) + tuple(shape[1:]))
        if e.store == 23:
            E, O, K = shape
            cpe, spe = O * K // 2, O * (K // 128) * 2
            p = H.payload(self.path, e)
            codes = np.frombuffer(bytes(p[lo * cpe: hi * cpe]), np.uint8).reshape(hi - lo, O, K // 2)
            sb = E * cpe
            scales = np.frombuffer(bytes(p[sb + lo * spe: sb + hi * spe]), "<f2").reshape(hi - lo, O, K // 128)
            su, sv = self.i4r_planes(name)
            return C.i4r_unrotate(C.i4r_decode_rot(codes, scales), su, sv)
        if e.store == 16:
            o, k = shape
            first, last = lo // 128 * 128, (hi + 127) // 128 * 128
            root = name[: -len(".weight")]
            su = self._side(root, ".suh")
            sv = self._side(root, ".svh")[first:last]
            p = H.payload(self.path, e)
            dec = T.decode(bytes(p[first * k // 2: last * k // 2]), (last - first, k), su, sv)
            return dec[lo - first: hi - first]
        if e.store == 5:
            return C.q4c_decode(self.raw(name), shape)[lo:hi]
        if e.store == 7:
            return C.q8g64_decode(self.raw(name), shape)[lo:hi]
        if e.store == 24:
            raise NotImplementedError(f"{name}: Q6G64 has no decoder")
        # raw planes (F16/F32/I64) use other store codes
        return _raw_plane(self.raw(name), shape, 0)[lo:hi]

    def i4r_planes(self, name: str):
        root = name[: -len(".weight")]
        if root + ".suh" in self.entries:
            su = self._side(root, ".suh")
        else:  # gate_up shares the layer's experts.su
            su = self._side(root.rsplit(".", 1)[0], ".su")
        sv = self._side(root, ".svh")
        return su, sv

    @functools.lru_cache(maxsize=4)
    def lm_head(self) -> np.ndarray:
        return self.get("lm_head.weight")


@functools.lru_cache(maxsize=1)
def tokenizer(path: str = "/home/quaternion/models/Signal-3.8-Flash-Next-HGN-v2-abliterated-4d30467d-20261014T203725Z/tokenizer"):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
