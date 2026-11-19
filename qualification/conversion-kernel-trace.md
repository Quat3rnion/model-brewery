# Optional converter GPU activity capture

Use a fresh evidence directory for each conversion. The commands below do not
acquire locks. Run them inside the approved GPU and quiet build locks on the
target host after checking memory, device use, and disk space. Profiling adds
overhead; do not use its timings as throughput measurements.

## CUDA (verified on the controller RTX 5070 Ti)

Build the installed NVIDIA CUPTI `cupti_trace_injection` sample as a
shared library in scratch, or use the same known collector from the
controller probe. The sample is not part of the converter. For the
toolkit layout tested on the controller:

```bash
# Run this compile under a build slot.
D=/opt/cuda/extras/CUPTI/samples
CUPTI_SO=/absolute/scratch/libcupti_trace_injection.so
g++ -O2 -std=c++17 -fPIC -shared -I/opt/cuda/include \
  -I"$D/common" -I"$D/common/nvtx" -o "$CUPTI_SO" \
  "$D/cupti_trace_injection/cupti_trace_injection.cpp" \
  "$D/common/nvtx/nvtx_payload_attributes.cpp" \
  "$D/common/nvtx/nvtx_payload_parser.cpp" \
  -L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64 -lcuda -lcupti
```

Use the device UUID to bind the converter to one physical GPU. This is an
operator wrapper, not a replacement for the converter executable in the
workflow protocol. Use distinct, nonexistent operator-selected evidence
and output directories, both outside the source tree.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" bash qualification/conversion-kernel-trace.sh \
  "$EVIDENCE_DIR" "$CUPTI_SO" -- "$CONVERTER" convert \
  --source "$SOURCE" --output "$OUTPUT" --profile v2 \
  --ht-backend cuda --i4r-backend cuda --q4cp-backend cuda
```

For `ht43`, select `--profile ht43` and omit `--i4r-backend cuda`.
The converter can select backends independently; check the counts for
each codec expected from the plan. Device preflight and decode kernels
do **not** count. The capture keeps the first 32 completed activity records
per codec, total observed record counts, at most 128 diagnostic lines,
the command and executable/collector hashes, device inventory, and status.
It captures the converter output rather than forwarding unmodified JSON.

Exit 0 means the converter succeeded and at least one known encoder
kernel was observed. Exit 3 means success without such a record; other
converter failures are passed through, even if some kernels ran.
Read `status.txt` to distinguish these cases. Counts need not be exhaustive
and are never proof that every tensor used a GPU. Keep the conversion
manifest and independent verifier report as separate evidence. Only apply
this parser to the trusted converter and documented collector format.

## HIP (recipe only; not tested on Frame)

If the target host already has ROCprofiler-SDK, confirm the installed
`rocprofv3 --help` supports the switches below. Use the host-approved GPU
selector and resource locks. Do not install or bypass production admission
controls to obtain this trace.

```bash
mkdir -- "$EVIDENCE_DIR" || exit 1
sha256sum "$CONVERTER" > "$EVIDENCE_DIR/converter.sha256"
rocprofv3 --version > "$EVIDENCE_DIR/profiler-version.txt" 2>&1
rocminfo > "$EVIDENCE_DIR/rocminfo.txt" 2>&1
status=0
rocprofv3 --kernel-trace --output-format csv --output-directory "$EVIDENCE_DIR" \
  -- "$CONVERTER" convert --source "$SOURCE" --output "$OUTPUT" \
  --profile v2 --ht-backend hip --i4r-backend hip \
  > "$EVIDENCE_DIR/stdout" 2> "$EVIDENCE_DIR/stderr" || status=$?
printf 'profiled_command_exit=%s\n' "$status" > "$EVIDENCE_DIR/status.txt"
```

Record the exact argv, host, selected device, driver and conversion manifest
with the trace. Inspect its kernel CSV for completed device dispatches of
`encode_tiles` (HT3/HT4) and `i4r_quantize` (I4R when selected), rather than
HIP runtime calls or decoder dispatches. For `ht43`, omit the I4R flag.
Check space for the profiler CSV before a full run; it is not capped by
the command above. A successful command without attributable encoder
dispatches does not establish HIP encoding.
