# Stage 3: SafeTensors merge acceptance

These tests do not qualify a release. They use only CPU. Do not run the
acceptance script against an existing controller.

## Mergekit reference

Use a separate Python 3.12 environment. Fetch the pinned tools once:

```bash
export TMPDIR=/arboretum/tmp
export UV_CACHE_DIR=/arboretum/tmp/w10-mergekit/cache
uv venv --python 3.12 /arboretum/tmp/w10-mergekit/venv
uv pip install --python /arboretum/tmp/w10-mergekit/venv/bin/python \
  'torch @ https://download.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'mergekit==0.1.5'
```

Run the fixtures without network access:

```bash
UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /arboretum/tmp/w10-mergekit/venv/bin/python qualification/mergekit_reference.py
cargo test -p brewery-worker --lib weight_source::scratch
cargo test -p brewery-controller --test admin scratch::
```

The generator runs the mergekit tensor kernels. The Rust test writes small
SafeTensors files, runs the production merge, and compares the F32 values to
the pinned outputs. The absolute tolerance is `1e-6`. The two DARE fixtures
use `p=0` and must match byte for byte. With `p>0`, mergekit uses PyTorch
Bernoulli masks and rescales to the observed L1 norm. Brewery uses seeded
SplitMix64 and scales by `1 / (1 - p)` instead. The Rust tests verify replay,
different seeds, survivor counts, and survivor values for both DARE methods.
At equal trimming magnitudes, Brewery keeps lower flat indices first.

## Isolated Qwen test

Use the pinned CPU worker tools, Qwen2.5-0.5B-Instruct files, and check
fixtures from [Stage 1](weight_space_acceptance.md). Build the controller,
worker, dispatcher, and engine from this revision into an isolated bin
directory. Run:

```bash
python qualification/weight_space_merge_acceptance.py \
  --root /arboretum/w10-accept-new --bins /arboretum/tmp/w10-bins
```

The script starts a controller on a free loopback port. It registers the
pinned model and makes two small edits to the final norm in separate scratch
children. It merges those children with task arithmetic, TIES, DARE-linear,
and DARE-TIES through MCP. It checks tensor differences against the base and
runs the W4 integrity, load and generation, target and locality, and
perplexity checks. It saves the results in `evidence/report.json` and removes
the isolated model files. It does not run GSM8K or publish to the Library.
