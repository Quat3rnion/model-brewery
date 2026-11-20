# ROCmFPX GGUF qualification

## Scope

The supported encoder is the [charlie12345/ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) at commit `fb08d7cdb67082ddda15a25e64dc7ec413afc41c`.
The five presets below use private tensor types from this pinned fork.
Build `llama-quantize` and `llama-simple` from the same revision.
Files with private ROCmFPX types require a matching reader from that revision.
The staged receipt sets `stock_runtime_compatible` to `false`.

The CPU smoke uses a synthetic, one-layer Qwen2 GGUF.
It does not establish GPU support, HGN output, model quality, or support for other architectures.

## Supported formats

| Preset | GGML tensor type ID | `general.file_type` | Bytes per 32 weights |
| --- | ---: | ---: | ---: |
| `Q3_0_ROCMFPX` | 104 | 112 | 14 |
| `Q4_0_ROCMFP4` | 100 | 100 | 18 |
| `Q4_0_ROCMFP4_FAST` | 101 | 103 | 17 |
| `Q6_0_ROCMFPX` | 102 | 110 | 26 |
| `Q8_0_ROCMFPX` | 103 | 111 | 33 |

The tensor type ID and `general.file_type` are separate identifiers.

## Presets and exact overrides

Choose a ROCmFPX preset when you want the fork format as the base preset.
The fork routes eligible tensors according to its preset rules.
The controller records `encoder_family=rocmfpx` and the pinned `encoder_revision` for these selections.

Use a stock preset and an exact tensor-name override to create a mixed profile.
The CPU smoke uses `Q8_0` with `blk.0.attn_q.weight` set to `Q4_0_ROCMFP4_FAST`.
That output keeps `general.file_type=7`; only the named tensor uses private type ID `101`.
Enter an exact tensor name, not a regular expression.

## Worker requirements

A ROCmFPX worker needs both `gguf_output` and the separate `gguf_rocmfpx_output` capability.
The standard `gguf_output` capability alone does not admit ROCmFPX jobs.

| Setting | Requirement |
| --- | --- |
| `BREWERY_GGUF_ROCMFPX_QUANTIZER` | Absolute path to a regular, executable ROCmFPX `llama-quantize` binary. Use a separate path from `BREWERY_GGUF_QUANTIZER`. |
| `BREWERY_GGUF_ROCMFPX_SHA256` | Lowercase SHA-256 digest with 64 hexadecimal characters. |
| `capabilities.gguf_rocmfpx_output` | Set to `true`. Also set `capabilities.gguf_output` to `true`. |

The engine checks the configured digest before it uses the quantizer.
It copies the checked binary into private job work.
The worker accepts `inspect`, `plan`, `convert`, and `verify` jobs for the pinned family and revision.
It rejects a different revision or a binary with a different digest.

## CPU build and smoke check

Run these commands from the Model Brewery repository root.
Use a new working directory under `/arboretum` for each run.
The CMake settings match the recorded CPU build.
They disable shared libraries, CUDA, HIP, Vulkan, and native CPU tuning.

```sh
WORK=/arboretum/rocmfpx-cpu-smoke
SRC="$WORK/src"
BUILD="$WORK/build"
RUN="$WORK/smoke"
mkdir -p "$WORK"
test ! -e "$SRC" && test ! -e "$BUILD" && test ! -e "$RUN"

git clone --no-checkout https://github.com/charlie12345/ROCmFPX.git "$SRC"
git -C "$SRC" checkout --detach fb08d7cdb67082ddda15a25e64dc7ec413afc41c

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$BUILD" --target llama-quantize llama-simple --parallel 2

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install "$SRC/gguf-py"
PYTHONPATH="$SRC/gguf-py" "$WORK/venv/bin/python" -B qualification/rocmfpx_smoke.py \
  --quantizer "$BUILD/bin/llama-quantize" \
  --reader "$BUILD/bin/llama-simple" \
  --out "$RUN"
```

The smoke creates an F32 fixture and checks dry-run plans, encoded layouts, and CPU decode.
It checks all five presets and one exact mixed override.
The recorded run decoded four tokens with the matching fork reader for each preset and the override.
The receipt records the source, quantizer, reader, and output digests.

## Stage a worker proposal

`deploy/lan/stage_local_gguf.py` stages the separate quantizer, sets the ROCmFPX capability, and probes conversion and verification.
It writes a worker config and `dispatch.proposed.json`; it does not activate the proposal.
Use a new staging root and a tiny, unquantized GGUF fixture.

```sh
python3 deploy/lan/stage_local_gguf.py \
  --root "$STAGE" \
  --release "$BINS" \
  --quantizer "$STOCK_QUANTIZER" \
  --catalog crates/controller/src/gguf_catalog.json \
  --dispatch "$DISPATCH" \
  --rocmfpx-quantizer "$BUILD/bin/llama-quantize" \
  --rocmfpx-sha256 593e59cb7bcdd51ca4f9e7cac3ea42797c48dd1c18ac9b13552dd8c382ae3708 \
  --rocmfpx-probe-source "$TINY_GGUF"
```

The stager accepts the approved CPU quantizer digest shown above.
It runs its probe before it adds `gguf_rocmfpx_output` to the proposed dispatch.
Review and approve the proposal through the normal worker rollout process.
