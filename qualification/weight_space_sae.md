# W11 CPU SAE procedure

This check uses Qwen2.5-0.5B-Instruct BF16 at layer 16, role `l_out`. It runs
an isolated loopback controller with a new database and one CPU worker. It
does not use the live controller. It does not change model weights or
require the W4 check bundle. Results are **not release-qualified**.

## Inputs and training

- Use the GGUF and Lanternfish pin in `weight_space_sae.py`.
- The harness downloads four pinned first chapters from
  [Standard Ebooks](https://standardebooks.org/about). The source texts
  are in the US public domain. Standard Ebooks releases its changes under
  [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
  The harness uses Carroll, Wells, and Dickens for training. It holds
  out Shelley in full. It checks each chapter hash before use.
- The harness reads F32 activations from bounded W1 trace jobs. It checks
  each trace receipt and row hash. No bulk export or runtime change is
  necessary.
- Training uses a pinned CPU PyTorch tool in a separate environment.
  PyTorch supplies the gradient, Adam optimizer, and matrix kernels.
  The controller does not require Python or PyTorch. SAE reads and
  interventions use Rust and the pinned trace runtime.
- The target is held-out explained variance of at least 0.60 with 1,024
  features and at most 48 active features per token. The trainer reports
  held-out MSE in raw activation units, explained variance
  `1 - MSE / centered_variance`, and mean held-out L0. A feature is dead
  if it never activates on a training row.

Use the shared build target and the build gate:

```sh
export TMPDIR=/arboretum/tmp CARGO_TARGET_DIR=/arboretum/brewery-w3-target
build-gate.sh build cargo build --locked \
  -p brewery-controller -p brewery-worker -p brewery-engine
```

Use a new directory for each run:

```sh
run=/arboretum/w11-sae-check
mkdir -p "$run"
export UV_CACHE_DIR="$run/uv-cache"
uv venv --python 3.12.15 "$run/venv"
uv pip install --python "$run/venv/bin/python" \
  --index https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
uv pip install --python "$run/venv/bin/python" \
  'safetensors==0.5.3' 'numpy==2.2.6'
PYTHONDONTWRITEBYTECODE=1 "$run/venv/bin/python" -m unittest discover \
  -s tools -p test_train_sae.py
PYTHONDONTWRITEBYTECODE=1 "$run/venv/bin/python" qualification/weight_space_sae.py \
  --bins "$CARGO_TARGET_DIR/debug" --inputs "$run/inputs" --evidence "$run/evidence"
```

To train a different SAE, save separate F32 `train` and `heldout` matrices
in one SafeTensors file. Supply a binding JSON file with the model ID,
GGUF SHA-256, layer, role, name, and corpus provenance. Run
`tools/train_sae.py --help` in that environment. The tool writes an
artifact only if the holdout reaches the requested reconstruction target.

## Public import

The harness imports the [Apache-2.0 ReLU SAE](
https://huggingface.co/HuggingAnalist/sae-qwen2.5-0.5B-res/tree/59535fa7275112a34424fe247028d2b8391c63c9/layer_16/final_sae)
for `Qwen/Qwen2.5-0.5B-Instruct`. Its SAELens SafeTensors file already uses
the same tensor names and shapes. Use it without conversion. Read its
`cfg.json` and supply Brewery binding metadata. The config uses
`model.layers.16` at block output with `apply_b_dec_to_input=true` and
no activation normalization. The config does not pin the source model
revision. The card and config identify different corpora. The local
binding and parity check do not verify the source training.

The harness compares feature IDs and activations with PyTorch on the same
captured rows. It steers the trained SAE's highest active feature at one
position and records the logit shift from `diff_traces`. This shows a
causal effect on the captured logits, not what the feature represents.
