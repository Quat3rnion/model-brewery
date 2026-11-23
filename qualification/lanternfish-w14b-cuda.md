# W14b standalone CUDA backward pin

This pin is **not release-qualified**. It is not a replacement for the W0
trace pin or the W9 managed edit pin. The W9 receipt parser does not accept
the new `provenance` field.

Lanternfish commit: `3b59809c42e8301222ca4a2bc3121266c12a3532`.
llama.cpp base: `184f6536024668e8908ab8c86033d56708e62b40`, with the eight
patches from that Lanternfish commit. The CUDA build uses nvcc 13.5.37,
GCC 16.2.1, `GGML_NATIVE=OFF`, `GGML_CUDA=ON`, `GGML_CUDA_FA=OFF`,
`GGML_CUDA_NCCL=OFF` and `CMAKE_CUDA_ARCHITECTURES=120` (resolved to 120a).
The binary uses the release profile and no default Cargo features.

To check and select the copied runtime libraries on the test host:

```sh
pin=/arboretum/pins/lanternfish-w14b-3b59809
(cd "$pin" && sha256sum -c /arboretum/brewery-w14b/qualification/lanternfish-w14b-cuda.sha256)
export LD_LIBRARY_PATH="$pin/lib"
```

The checksum file covers the binary and all five project libraries. It
does not cover system libraries, the NVIDIA driver, or the CUDA toolkit.
Use a GPU queue ticket for model loads. Both `grad` and `locate-edit`
accept `--backend cuda`; CPU remains the default. The factor solve and
host-to-device transfers do not count as ggml graph operations. The
native call rejects CPU computation in the forward and backward passes.
Patch 0007 checks the scheduled backend before each graph dispatch. The
CLI places the input embedding table on the GPU so `GET_ROWS` also runs there.

## Results and limits

Model: Qwen2.5-0.5B-Instruct F32, SHA-256
`8df5d93dbefb5f73ce905c6e464aaa312756fab7a8a85ee1f7dae15cec36bfde`.
GPU: RTX 5070 Ti, driver 620.73.04. Use the acceptance harness from the
Lanternfish commit to reproduce the primary check from that checkout:

```sh
export TMPDIR=/arboretum/tmp
resource-queue run --resource gpu --holder w14b-check --max-hold 300 --wait 300 -- \
  resource-queue run --resource build --exclusive --holder w14b-quiet --max-hold 240 --wait 60 -- \
  python3 -B tools/w14b_acceptance.py --binary "$pin/lanternfish" \
    --oracle /arboretum/brewery-w14b/qualification/lanternfish-w14b-oracle.json \
    --oracle-sha256 caf1508dcdcbe5c060e058a88f6759796c188f1e7861a7d0bcd281e6fd328a0e \
    --threads 6 --evidence /arboretum/w14b/check-new
```

Use a new evidence directory. Acceptance requires the held-out tensor
relative error, tiny finite differences and the pinned FP64 projection
oracle. The CPU pointwise comparison is a separate diagnostic, not a gate.
The copied-pin report is `/arboretum/w14b/acceptance-strict/report.json`
(SHA-256 `a8a3fbc7e35829ff10d82857035282b55e4aae61b80b720cfc7174c72526f9b9`).
All 13 CLI tests and nine harness and placement tests passed.

- With only GPU admission enabled, `GGML_SCHED_DEBUG=2` showed one CPU
  `ACC` on the two-layer model and 23 on the 24-layer model for a layer-0
  gradient. Both weight and residual backward needed the strided query
  gradient. Patch 0004 keeps these operations on the GPU. Existing
  `OUT_PROD`, `RMS_NORM_BACK`, `SILU_BACK`, `ROPE_BACK`, `SOFT_MAX_BACK`,
  `REPEAT_BACK`, `GET_ROWS_BACK` and `CROSS_ENTROPY_LOSS_BACK` did not
  need new kernels.
- Tiny Qwen2: maximum absolute finite-difference error `3.67e-5` or
  less (limit `3e-4`, centered step `0.003`). For the real model, the
  maximum error divided by the peak absolute CPU gradient was `6.20e-5`,
  `1.73e-4`, and `4.10e-4` at layers 0, 12, and 23. The separate pointwise
  limit `1e-5 + 2e-3 * abs(CPU)` **failed** at all three layers. Six
  held-out prompt/layer pairs also failed the pointwise limit, but all
  six passed a separately declared tensor-relative limit of `1e-3`.
  The worst held-out tensor-relative error was `3.02e-4`; the worst
  pointwise error was 9.32 times the diagnostic limit on the held-out set.
- An FP64 dot product on the captured output projection operands showed
  a maximum CPU error of `8.08e-6` and a maximum CUDA error of `7.36e-7`
  for this intermediate. The reduction spans 151936 vocabulary rows.
  This is not an FP64 oracle for the full gradient.
- W9 ROME factor calculation: CounterFact case 7, layer 12, 16 steps,
  one warm-up and three measured runs per backend, fresh moment cache each
  time. CPU median 6.626 s (range 6.626 to 7.127); CUDA median 2.167 s
  (2.167 to 2.767). Speedup: 3.06x. This is CLI wall time, not kernel
  time or a managed Brewery job. GPU clocks were not locked. In a separate
  ROME placement run, 20 forward graphs (12120 compute operations) and 44
  training graphs (41712 compute operations) ran on CUDA0. The 11 `grad`
  runs covered 8724 compute operations. The CPU compute count was zero.
- With the isolated ROCm 10.1 SDK and Clang 23.1.5, the HIP library compiled
  and linked for both gfx1036 and gfx1151. A native strided `ACC` check
  with shape `(9, 3, 17)` passed exactly on the local gfx1036 Radeon 610M.
  The tiny full weight-gradient probe did not pass on that GPU: the audit
  assigned 95 operations to ROCm0, then the ROCm runtime reported a memory
  fault in a Tensile GEMM kernel. Testing stopped. No gfx1151 runtime
  was available. The CLI accelerator selector and this pin are CUDA-only.
- `llvm-objdump` reported eight VOPD pairs in the default gfx1151 `ACC`.
  Two have distinct same-parity SRC0 registers, a known interlock risk.
  The kernel used 20 VGPRs, 46 SGPRs, zero spill slots and no LDS. A
  diagnostic compile with `-mllvm -amdgpu-enable-vopd=0` eliminated the
  pairs. This is ISA evidence, not a gfx1151 execution result.
- The CUDA strided `ACC` streaming probe moved 177995776 logical bytes in
  250.14 us median (246.76 to 411.71) for `(64, 14, 16384)`. That is
  711.6 GB/s, or 79.4% of the nominal 896 GB/s memory-bandwidth bound.
  This is not the end-to-end edit speed. The original kernel did not support
  this input layout. The run used 10 warm-ups and nine batches of 80 calls;
  see `/arboretum/w14b/acc-check.cpp` and `acc-check.log`.
