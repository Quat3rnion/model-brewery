# Make GGUF quantization variants

Use this procedure to make named GGUF files from one unquantized GGUF.
Use the same source for all variants in a comparison. Do not use an
already quantized GGUF or a split GGUF as the source.

## Requirements

- Python 3.11 or later and a trusted `brewery-engine` binary from this repository.
- An approved stock CPU `llama-quantize` binary and its approved SHA-256 digest.
  Use a binary that does not need libraries next to it. The driver makes a
  private copy and checks its digest before it starts the engine.
- An existing parent directory with sufficient space for all output files.
  Select a new output directory for each run.

Conversion is CPU-only. The driver uses the existing engine, which clears GPU
visibility for the quantizer. Do inference and quality checks on `frame` under
a GPU lease. The driver does not load a model for inference or submit a job.
Get operator approval before you convert a full model.

## Convert

Save the variant definitions as `variants.json`:

```json
{
  "q8": {
    "gguf_type": "Q8_0",
    "threads": 2
  },
  "q4k-embed-f16": {
    "gguf_type": "Q4_K_M",
    "tensor_types": {"token_embd.weight": "F16"},
    "threads": 2
  }
}
```

Each key becomes a file name. Use one to 64 characters: lowercase letters,
digits, hyphens, or underscores. Start with a letter or digit. `gguf_type`
selects a stock GGUF preset. The optional `tensor_types` map selects types for
exact tensor names. The engine rejects unknown names and unsupported types.
If you omit `threads`, the engine uses its default CPU budget. If a variant
needs an importance matrix, add `--imatrix /path/to/imatrix`. The same matrix
applies to all variants in the run.

Run from the Model Brewery repository root:

```sh
python3 -B tools/gguf_variants.py \
  --engine "$BREWERY_ENGINE" \
  --quantizer "$CPU_LLAMA_QUANTIZE" \
  --quantizer-sha256 "$APPROVED_QUANTIZER_SHA256" \
  --source "$SOURCE_GGUF" \
  --variants variants.json \
  --out /arboretum/gguf-variants-run
```

The engine makes a dry-run plan before each conversion. It checks the output
against that plan. When all checks pass, the driver writes `<name>.gguf` for
each variant and prints the path to `manifest.json`. If a check fails, the
driver does not write a manifest. Use a new directory for a retry.

Keep the manifest with the GGUF files. It records the physical source,
quantizer, and output SHA-256 digests. It also records the matrix digest if
used. For each output it records the preset, requested tensor types, CPU
thread count, and verification status. Output paths in the manifest are
relative to the manifest directory. File verification does not measure
inference quality and does not qualify a release.
