# Native packed ternary GEMV

`gemv.h` exports a separate launch function for each format on each GPU API.
For example, `brewery_hip_tq1_0_gemv` reads stock TQ1_0 rows (ID 34, 54 bytes
per 256 weights); `brewery_cuda_tq2_0_gemv` reads stock TQ2_0 rows (ID 35,
66 bytes per 256 weights). Both take one FP32 input vector and write FP32
outputs. The GPU decodes packed bytes in registers. The kernel object does
not link to ggml. No GGUF reader, format guessing, ggml graph registration,
model inference, or fallback is part of this test.

## Numeric evidence

The CPU oracle is the stock ggml build at
`184f6536024668e8908ab8c86033d56708e62b40`. It encodes exact ternary
weights with different scales per block, including zero blocks. It decodes
the same bytes that the GPU receives, checks every weight against the
original, and computes each reference dot product in FP64. The CPU-only
check compares that dot to a sequential FP32 dot on the original weights.
The device test reads results back after synchronization. The bound for each
output is `1e-5 + 2e-6 * sum(abs(decoded_weight * activation))`.
The launcher also rejects a width that is not divisible by 256.

Maximum absolute error (2026-11-20). Both types used the same representable
matrix and activations, so their errors matched:

| Rows x columns | CPU, each type | CUDA sm_120, each type | HIP gfx1036, each type |
|---|---:|---:|---:|
| 1 x 256 | 3.38768587e-08 | 2.57277861e-08 | 2.57277861e-08 |
| 7 x 768 | 1.10356923e-05 | 3.79353878e-06 | 3.79353878e-06 |
| 65 x 4096 | 2.16320750e-04 | 2.31376835e-05 | 2.31376835e-05 |

The CUDA device was an NVIDIA GeForce RTX 5070 Ti. HIP reported the gfx1036
iGPU as "AMD Ryzen 7 9800X3D 8-Core Processor". Each device returned all
73 outputs per type within the bound. gfx1151 was **compiled and linked
only**; its HIP object contained the `hip-amdgcn-amd-amdhsa--gfx1151`
bundle. No gfx1151 runtime or performance result is available. No ggml
backend result changed. The red test failed to link the two missing CUDA
launchers before implementation.

## Reproduce on this host (Bash, from `native/gguf-ternary`)

```bash
set -euo pipefail
export OUT="$(mktemp -d /arboretum/ternary-native.XXXXXXXX)"
export TMPDIR="$OUT" BUILD_GATE_NAME=gguf-tq-stock BUILD_MAX_HOLD=90
export GGML_INCLUDE=/arboretum/llama.cpp/ggml/include
cpu_build=/arboretum/brewery-cpu-llama.WOcBfOjw
export GGML_LIB="$cpu_build/build/ggml/src"
test "$(cat "$cpu_build/revision")" = 184f6536024668e8908ab8c86033d56708e62b40
source /arboretum/rocm-sdk-10.1.misYj9Zo/env.sh
for arch in cpu sm_120 gfx1036 gfx1151; do
    /arboretum/gpu-queue/build-gate.sh build bash build.sh "$arch"
done
"$OUT/cpu"
for arch in sm_120 gfx1036; do
    /arboretum/gpu-queue/gpu-turn.sh gguf-tq-stock 60 "$OUT/$arch" "$arch"
done
llvm-objdump --offloading "$OUT/gfx1151.o"
```

Run logs: `/arboretum/ternary-native.abGSz09O`. The oracle's
`libggml-base.a` SHA-256 is
`977e0180c9c034b81c0353f0e32a65e8c424e9fe0fe72caba9b473ff77942aa7`.
Compiler versions: CUDA 13.4.92, system clang 23.1.5, staged ROCm SDK
10.1.0 (HIP 7.16.0). No Python or Torch runtime is used.
