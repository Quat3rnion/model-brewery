# Native packed operations

These checks apply to matrix operations, not complete models. The tests compare
outputs from packed weights with independent decoded results or pinned vectors.
None of these kernels needs Python or PyTorch to run.

| Directory under `native/` | Operation tested on CPU, NVIDIA `sm_120` and AMD `gfx1036` |
| --- | --- |
| `hgn` | BF16, F32, F16, Q4C-P, Q8G64, Q6G64, HT3, HT4 and I4R matrix-vector products. Includes HT3/HT4 and I4R expert layouts with rotation sidecars. FP8G uses a row gather; I64 is host metadata, not a GPU contraction. |
| `gyro` | Agention wire IDs 144 through 151. Both original and rotated inputs were tested with signed rotation and identity rotation. The build checks the upstream codebook before use. |
| `trellis-ref/glq*` | GLQ 3INST, 2 through 8 bits per weight, with a batch of three 32-by-64 products per rate and one 96-by-96 case at 5 bits. Maximum absolute GPU error: `1.05e-5`. |
| `trellis-ref/qtip*` | 31 Cornell QTIP cases, including reference and swizzled layouts. Maximum absolute error: `4.77e-7`. The public interface specifies the supported axes, scale and sidecars. |

The `gfx1151` builds for these operations were cross-compiled, not run.
Use a matching device for an execution check. From the repository root, for
example, run `make -C native/hgn cpu` then `native/hgn/.build/probe-cpu cpu`.
For QTIP, build `qtip.cpp` and `qtip_test.cpp` together with a C++17 compiler,
then run the result. Those tests use the included vectors; the Python oracle
scripts are for regenerating vectors, not for running the kernels.
These results do not enable a worker, verify model generation, or qualify a
conversion for release.

# Derived FP32 replay experiment

This experiment is separate from native GGUF qualification. Do not use
its results in the native gate. The original model digests, token arrays,
receipts, and comparator tolerances do not change.

`reference_f32.cpp` has two opt-in operations:

- `op`: dequantize the saved Granite layer-0 Q matrix from Q4_K to F32.
  Run one F32 matrix-vector product on CPU and CUDA, using the same
  stored activation and matrix on both. Write both results and an FP64
  dot-product reference for every row.
- `convert`: write a new GGUF with every tensor materialized in F32.
  This preserves the numerical values represented by the quantized
  weights; it does not recover the original unquantized checkpoint.
  The new file has a distinct digest and model name and records the
  source digest in GGUF metadata. It cannot overwrite a file.

`reference_replay.py` verifies the pinned source, creates the derivative,
and uses Lanternfish's existing fixed-token replay. It verifies the
derivative's digest in each receipt, checks completed GPU matmuls at
every GPU step, and applies the unchanged comparator. It emits a
`brewery.reference-replay.v1` envelope, not a native replay receipt.
An explicit `--gpu-layers` sets mixed CPU/CUDA placement; it does not
qualify full offload. The default refuses admission if available VRAM
cannot hold the derivative plus 1 GiB of headroom.

The model-level mode uses F32 stored weights, CUDA cuBLAS FP32, disabled
TF32, and disabled CUDA fusion. It retains the pinned runtime's ordinary
non-matmul operations and KV cache types. It is not an end-to-end FP64
reference or a guarantee of bitwise CPU/CUDA arithmetic.

## Local reproduction

Run from the Brewery root on the local RTX 5070 Ti. Use a new diagnostic
run directory for each invocation.

```sh
L=/arboretum/llama.cpp
R=/arboretum/lanternfish-brewery/reference
A=/arboretum/lanternfish-brewery/diagnose/granite-l0
export PATH="$HOME/Projects/agent-resource-guarding/bin:$PATH"
export TMPDIR="$R/t" LD_LIBRARY_PATH="$L/build/bin:/opt/cuda/lib64"
export BUILD_GATE_NAME='lanternfish-reference' BUILD_MAX_HOLD=40 QUIET_MAX_HOLD=185
mkdir -p "$R/t" "$R/run"
build-gate.sh build c++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off \
  -I"$L/ggml/include" qualification/reference_f32.cpp -L"$L/build/bin" \
  -Wl,-rpath,"$L/build/bin" -lggml -lggml-base -o "$R/reference_f32"
NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=fp32 \
  gpu-turn.sh "$BUILD_GATE_NAME" 45 build-gate.sh quiet \
  "$R/reference_f32" op "$A/cpu-q-x.f32" "$A/q-original.gguf-weights" \
  >"$R/op.tsv" 2>"$R/op.log"
gpu-turn.sh "$BUILD_GATE_NAME" 200 build-gate.sh quiet \
  python3 qualification/reference_replay.py moe \
  --source /arboretum/brewery-inputs/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf \
  --out "$R/run/moe-new" --device CUDA0 --converter "$R/reference_f32" \
  --lanternfish /arboretum/lanternfish-target/release/lanternfish \
  --comparator /arboretum/model-brewery-trace-parity/target/debug/trace-parity
```

For a bounded dense comparison on this host, select `dense`, source
`$HOME/models/outpost/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf`, a fresh output
directory, and `--gpu-layers 16`. For full offload, omit `--gpu-layers`
only on a host with sufficient admitted VRAM.

## Observed result on RTX 5070 Ti

- **Granite Q projection:** CUDA completed one F32 matmul in its
  `CUDA0` buffer. Across all 1,024 rows, CPU/CUDA max absolute error
  was `2.48e-5`, versus `1.08594418` with the native quantized inputs to
  the matmul. Max error against FP64 was `2.18e-5` on CPU and `1.59e-5`
  on CUDA.
- **Derived Granite F32, all layers offloaded:** the unchanged
  comparator exited 1. The first failure was residual index 281 at
  layer 12, position 0: `0.0122659653 > 0.0114224511`. Six ordered
  expert-ID rows also differed. The GPU replay observed 241 CUDA
  matmuls at each step; the load log reports 25/25 layers offloaded.
- **Derived Qwen F32, mixed CPU/CUDA:** the unchanged comparator
  exited 0 with 160/160 residual, attention, and FFN rows. CUDA
  ran 125 observed matmuls at each step, and the load log shows 16/33
  layers offloaded. This is a result for the derivative on mixed
  placement only, not an all-GPU result or a pass for the pinned GGUF.
- **Derived Qwen F32, all layers offloaded:** not run. Admission
  refused with 13,410 MiB free. The derivative was 16,833,973,568
  bytes. Even subtracting the input embedding tensor leaves
  14,280,208,384 bytes of F32 tensor storage, greater than the
  14,061,404,160 bytes of free VRAM before buffers.

Evidence is under `/arboretum/lanternfish-brewery/reference/`:
`op.tsv`, `op.log`, `hardware.csv`, `dense-full-admission.log`, and
`run/{moe,dense-mixed}/reference-result.json` with adjacent runtime logs.
The output envelopes preserve the source pin and the distinct derivative
digest. The native receipts remain unchanged and are not qualified by
this experiment.

## Native runner on the unchanged GGUFs

`qualification/replay_host.py` captured both pairs on the RTX 5070 Ti in
one admitted run. At each step, the dense replay recorded 265 CUDA matmul
nodes and Granite recorded 241; neither recorded a CPU matmul. Both failed
at the first logit. The dense difference was 0.597267091 (bound 0.0662254715);
Granite's was 1.15857029 (bound 0.171904583). The traces, comparison output
and hashes are in `/arboretum/brewery-fixed-native.iRv7jjoH/run/failed.json`
and its directory. This is a failed gate, not an incomplete capture.

At pinned llama.cpp revision `184f6536024668e8908ab8c86033d56708e62b40`,
the CPU Q4_K and Q6_K dot-product paths quantize activations to Q8_K.
The CUDA matrix-vector path used here takes Q8_1. The CUDA dispatch
in `ggml/src/ggml-cuda/ggml-cuda.cu` selects this path before the matrix-
matrix or cuBLAS selection. The `GGML_CUDA_FORCE_MMQ` and
`GGML_CUDA_FORCE_CUBLAS` build options do not switch this replay to a
common activation format. A compatible arithmetic implementation would
be a new runtime build. It would need its own pin and full validation.

## Isolated modified runtime on sm_120

The opt-in Q8_K candidate passed all 49,155 first-position Granite logits,
but failed the unchanged full replay on both models. The subsequent CUDA
normalization trial matched the CPU on all 3,072 measured layer-0
normalization outputs. The next projection still differed by up to
`1.52588e-5`. With identical inputs, standalone FP16 cache writes matched
bit for bit on all 5,120 measured K/V values per origin. This does not test
the fused RoPE/write path.

The release comparator still rejected Granite at position 1 and Qwen at
position 0. Each GPU replay observed only GPU matmuls. Neither trial is
deployed or qualified. Patches, receipts, and reproduction commands:
`/arboretum/brewery-cuda-q8k.U9qywp/{RESULT,MATCHED-KV}.md`.
