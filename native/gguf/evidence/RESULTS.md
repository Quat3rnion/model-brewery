# Native GGUF matvec evidence

The input is `tensor_types` in `crates/controller/src/gguf_catalog.json`, not
the quantizer preset list. The stock pin has 27 types. The ROCmFPX pin has
31 types: 26 shared with stock and five fork-only types (100-104). `Q2_0`
(42) is stock-only. The union is 32 distinct types.

The C++ probe checks the type name, numeric ID, block size, and bytes per
block. It packs three deterministic fixtures of 33x512, 65x1024, and 97x1536
weights. It sends the exact packed weight tensor and an F32 vector to one
`GGML_OP_MUL_MAT` node. On GPU, all tensors must be in non-host buffers on
the selected device. The probe calls that backend directly, without a ggml
scheduler or CPU backend fallback. It uses the pinned host format decoder
and an independent scalar FP64 dot product as the reference. Each output
must be finite and differ by no more than `1e-4 + 1e-4 * sum(abs(w_i*x_i))`.
`max_tolerance_ratio` is the worst absolute error divided by that limit.

This tests packed-input backend execution, not the choice of GPU instruction
or whether ggml fuses dequantization and multiplication internally. It does
not qualify GGUF file parsing, models, or performance.

## Local results

| Pin | Backend | Passed | Failed | Note |
| --- | --- | ---: | ---: | --- |
| Stock | CPU | 27 | 0 | Worst tolerance ratio 0.000226 |
| ROCmFPX | CPU | 31 | 0 | Worst tolerance ratio 0.000226 |
| Stock | CUDA, RTX 5070 Ti (SM120) | 25 | 2 | `TQ1_0`, `TQ2_0` are rejected by `ggml_backend_cuda_device_supports_op` for `MUL_MAT`; no CPU retry. Worst passing ratio 0.413. |
| ROCmFPX | CUDA, RTX 5070 Ti (SM120) | 29 | 2 | All five fork-only types pass. The same two ternary types are rejected; no CPU retry. Worst passing ratio 0.413. |
| Stock | HIP, gfx1036, baseline | 2 | 25 | BF16 and F16 pass; 23 numerical failures and two unsupported ternary types. |
| ROCmFPX | HIP, gfx1036, baseline | 2 | 29 | BF16 and F16 pass; 27 numerical failures and two unsupported ternary types. |
| Stock | HIP, gfx1036, diagnostic barrier | 25 | 2 | Only `TQ1_0` and `TQ2_0` are unsupported. |
| ROCmFPX | HIP, gfx1036, diagnostic barrier | 29 | 2 | All five fork-only types pass; only `TQ1_0` and `TQ2_0` are unsupported. |

The TSV files contain one row per applicable tensor type and backend. A failed
row does not count as execution. The process exits nonzero if any row fails.
Negative checks with `.build/stock-cpu` returned exit 1 for a CUDA request
(`missing_backend_registry`), a wrong CPU index (`missing_backend_device`),
and a wrong packed byte count in the catalog argument (`wrong_block_layout`).
The stock probe rejected fork-only type 100; the fork probe rejected ID 42
when requested as stock `Q2_0`.

The staged ROCm SDK at `/arboretum/rocm-sdk-10.1.misYj9Zo` has a working core
runtime on `gfx1036`; `smoke.log` records a separate HIP kernel launch. The
first CMake configuration for each pin found Clang 23.1.5 but stopped at
`ggml/src/ggml-hip/CMakeLists.txt:47 (find_package)` because hipBLAS was
absent. After signed hipBLAS and rocBLAS packages were staged into the
scratch SDK, configuration succeeded for both pins. The shared CUDA/HIP
`MUL_MAT` type lists at both pins also omit the two ternary types.

## HIP diagnosis on gfx1036

Both unmodified pins reported success for each accepted graph. Without a
workaround, however, the probe found 50 numerical failures across the two
pins. Each failed at the first output of the width-512 fixture, returning
zero instead of the reference in `hip-baseline-mismatches.tsv`. For `Q4_0`
on both pins: actual `0`, reference `-6.35940552`, limit `0.024781955`.
`stock-hip.tsv` and `rocmfpx-hip.tsv` preserve the baseline runs.

The one-variable `Q4_0` checks found:

- Disabling graph capture alone did not help. Copy serialization alone also
  did not help (`hip-q4-isolate.tsv`).
- `AMD_SERIALIZE_KERNEL=1` did not help. `=2` and `=3` each gave exact matches
  for all 195 outputs on both pins (`hip-q4-kernel-modes.tsv`). Mode 2 waits
  after kernel launch; mode 1 waits before it.
- In the independent `diagnostics/q8-barrier.cpp` test shim, no barrier or
  a barrier before `quantize_row_q8_1_cuda` left the failure unchanged. A
  barrier immediately **after** that launch gave exact matches for all 195
  `Q4_0` outputs on both pins (`hip-q4-barrier.tsv`). The pinned `mmvq.cu`
  submits this Q8_1 activation producer and the packed matvec consumer on
  the same stream. This establishes a synchronization-sensitive failure at
  their handoff on this stack. It does not establish which runtime,
  driver, or compiler component is responsible for the same-stream failure.

The repeated full matrix with `AMD_SERIALIZE_KERNEL=3` only
(`*-hip-kernel-serialized.tsv`) and with the narrower post-Q8 barrier
(`*-hip-q8-barrier.tsv`) produced identical results. In each case, the
same pinned device code, fixture, and oracle were used; no host fallback,
architecture override, or threshold change was used. The shim ran once
per fixture for all 50 quantized type/pin pairs. Both tests left the
two unsupported operations unchanged. Neither is a production fix or a
qualification of graph capture, tuned performance, other devices, or
matrix-matrix paths.

## Reproduce with the local pinned builds

From `native/gguf`:

```sh
stock=/arboretum/llama.cpp
stock_cpu=/arboretum/brewery-cpu-llama.WOcBfOjw
fork=/arboretum/brewery-rocmfpx-cpu.Gm2P4nXF
stock_pin=184f6536024668e8908ab8c86033d56708e62b40
fork_pin=fb08d7cdb67082ddda15a25e64dc7ec413afc41c

./build.sh stock cpu "$stock/ggml/include" "$stock_cpu/build/ggml/src" "$stock_pin"
GGUF_VERIFIED_SOURCE_REVISION="$fork_pin" \
  ./build.sh rocmfpx cpu "$fork/src/ggml/include" "$fork/build/ggml/src" "$fork_pin"
./build.sh stock cuda "$stock/ggml/include" "$stock/build/bin" "$stock_pin"
./run.sh stock cpu > evidence/stock-cpu.tsv
./run.sh rocmfpx cpu > evidence/rocmfpx-cpu.tsv
resource-queue run --resource gpu --holder gguf-matvec --max-hold 105 \
  --flock gpu -- ./run.sh stock cuda > evidence/stock-cuda.tsv
```

Run each command even when another returns 1. Use a ggml build with
the requested backend built in (`GGML_BACKEND_DL=OFF`). To run HIP when the
libraries are available, first use `build.sh {stock|rocmfpx} hip` with the
matching include and shared library paths, then run HIP under the GPU queue.
`run.sh` takes an optional registry device index as its third argument.

To configure, build, and run HIP with the staged SDK, use `family=rocmfpx`,
`src="$fork/src"`, and `pin="$fork_pin"` or use the stock equivalents:

```sh
source /arboretum/rocm-sdk-10.1.misYj9Zo/env.sh
export PATH="$ROCM_PATH/bin:$PATH" TMPDIR="$PWD/.build/tmp"
unset HSA_OVERRIDE_GFX_VERSION
family=rocmfpx src="$fork/src" pin="$fork_pin"
# For stock: family=stock src="$stock" pin="$stock_pin"
b=".build/$family-hip-build"
mkdir -p "$TMPDIR"
cmake -G Ninja -S "$src" -B "$b" \
  -DGIT_EXE=FALSE -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
  -DGGML_HIP=ON -DGGML_CUDA=OFF -DCMAKE_HIP_COMPILER=/usr/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" \
  -DAMDGPU_TARGETS=gfx1036 -DCMAKE_HIP_ARCHITECTURES=gfx1036 \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -Dhipblas_DIR="$ROCM_PATH/lib/cmake/hipblas" \
  -Drocblas_DIR="$ROCM_PATH/lib/cmake/rocblas" \
  -DGGML_NATIVE=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
resource-queue run --resource build --holder "gguf-$family-hip" --max-hold 540 --flock build -- \
  cmake --build "$b" --target ggml --parallel 3
GGUF_VERIFIED_SOURCE_REVISION="$pin" \
  ./build.sh "$family" hip "$src/ggml/include" "$b/bin" "$pin"
resource-queue run --resource gpu --holder gguf-matvec --max-hold 105 --flock gpu -- \
  ./run.sh "$family" hip > ".build/$family-hip-repro.tsv"
```

For the small isolation test and full matrices with the diagnostic shim,
reuse the pinned HIP probe binaries without changing their sources or output
records:

```sh
unset HSA_OVERRIDE_GFX_VERSION AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY
unset GGML_CUDA_DISABLE_GRAPHS LD_PRELOAD
c++ -I"$stock/ggml/include" -isystem "$ROCM_PATH/include" \
  -D__HIP_PLATFORM_AMD__ -shared -fPIC -O2 -Wall -Wextra -Werror \
  -o .build/q8-barrier.so diagnostics/q8-barrier.cpp -L"$ROCM_PATH/lib" -lamdhip64 -ldl
resource-queue run --resource gpu --holder gguf-matvec --max-hold 105 --flock gpu -- \
  bash -c '
    for f in stock rocmfpx; do
      for mode in none before after; do
        env LD_PRELOAD="$PWD/.build/q8-barrier.so" GGUF_Q8_BARRIER="$mode" \
          ".build/$f-hip" ROCm 0 Q4_0 2 32 18 0
      done
      env LD_PRELOAD="$PWD/.build/q8-barrier.so" GGUF_Q8_BARRIER=after \
        ./run.sh "$f" hip > ".build/$f-hip-q8-barrier-repro.tsv"
    done'
```

The loaded stock ggml binaries report `184f653`. The stock CPU build has a
full `revision` file, and its CMake source is `/arboretum/llama.cpp`.
The fork CMake cache points to `"$fork/src"`; `"$fork/src.tar.gz"` is rooted
at `ROCmFPX-fb08d7cdb67082ddda15a25e64dc7ec413afc41c/` and has SHA-256
`ddcda52aa6dafa2cf99109b428b6c3f3e37b23786b34cf78e47634b0d38cfa58`.
Its tarball build reports `unknown`; use `GGUF_VERIFIED_SOURCE_REVISION`
only after checking the source archive and build independently.

The fork CUDA build used the same unpacked source and CUDA 13.4.92. It
required one build-only workaround for CCCL 3.5: `top-k.cu` did not include
the definition of `cuda::discard_iterator`. Force-include `<cuda/iterator>`
only when building that object; no source files or matvec flags change:

```sh
mkdir -p .build/tmp
export TMPDIR="$PWD/.build/tmp"
b=.build/fork-cuda-top
cmake -G Ninja -S "$fork/src" -B "$b" \
  -DGIT_EXE=FALSE -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
  -DGGML_CUDA=ON -DGGML_HIP=OFF -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DCUDAToolkit_ROOT=/opt/cuda -DGGML_NATIVE=OFF -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_CUB_3DOT2=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
resource-queue run --resource build --holder gguf-fork-cuda --max-hold 540 --flock build -- \
  env NVCC_PREPEND_FLAGS='--pre-include cuda/iterator' \
  ninja -C "$b" ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/top-k.cu.o
resource-queue run --resource build --holder gguf-fork-cuda --max-hold 540 --flock build -- \
  cmake --build "$b" --target ggml --parallel 4
GGUF_VERIFIED_SOURCE_REVISION="$fork_pin" \
  ./build.sh rocmfpx cuda "$fork/src/ggml/include" "$b/bin" "$fork_pin"
resource-queue run --resource gpu --holder gguf-matvec --max-hold 105 --flock gpu -- \
  ./run.sh rocmfpx cuda > evidence/rocmfpx-cuda.tsv
```
