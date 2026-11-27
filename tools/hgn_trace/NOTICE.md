# HGN CPU trace source

`forward.py` adapts the layer-streamed CPU capture by
`model-install-and-ablite` (task `t_9afc1ddb`). The original `capture.py`
has SHA-256 `54709943233bbd4088ecc6a7c3892000c31a302487855ce0305940919372e645`.
The Brewery trace adapter writes Lanternfish trace schema v1.

`wsp/hgn.py` is an unchanged snapshot from the model-coding toolkit.
Its source did not have a Git revision. It and its unchanged Apache-2.0
dependencies are pinned by content below and included in the runtime hash.
No model-coding checkout or separate decoder build is needed on a worker.

```text
48e8c210cdb21393450fea6252226a8e0b85518f091baab1760c908bea3f5c4c  wsp/hgn.py
f9612777daf6d2b26bdf1dd8971e739fd6ddc3bc8ae47a06909f54b7752cee92  wsp/vendor/hgnenc.py
6abedcfda20c5f86bc7f5ad3ad3363be84a65e3c0f8b4b658d9e94f64855ec5a  wsp/vendor/hgnht.py
159402e40df7d6e3b037d0fc2e2050ab3a27d675d3e026f7f18889e014a93744  wsp/vendor/hgnlib.py
```
