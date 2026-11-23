# X4 corpus-loss generalization

This check measures transfer between two fixed corpora. It is not a release
qualification. It uses the X4c model pair, exchange rule, equal-size target,
and full-vocabulary evaluator. It does not change the production damage job.

## Result

The advantage transferred in both directions on this model and these
corpora. The corpus recipe beat uniform, both random-group medians,
five of six original controls, and all three exchange-matched controls
on **both** perplexity and mean KL in each fold. Original seed 3 beat
the corpus recipe on both metrics in both folds. The corpus recipe was
not the best recipe tested.

| Recipe | Dolly selects, W4 scores: PPL | Mean KL | W4 selects, Dolly scores: PPL | Mean KL |
| --- | ---: | ---: | ---: | ---: |
| BF16 reference | 81.12227 | 0 | 33.50838 | 0 |
| Uniform | 90.06855 | 0.375718 | 53.70331 | 0.571136 |
| **Corpus** | **86.36626** | **0.154259** | **36.31007** | **0.186088** |
| Original seed 1 | 90.07882 | 0.378534 | 54.19539 | 0.571694 |
| Original seed 2 | 112.99528 | 0.673466 | 69.71426 | 0.875180 |
| Original seed 3 | 85.58890 | 0.139994 | 35.68960 | 0.181538 |
| Original seed 4 | 104.63930 | 0.455617 | 60.80433 | 0.703501 |
| Original seed 5 | 88.71597 | 0.372978 | 52.34030 | 0.569742 |
| Original seed 6 | 86.86215 | 0.376720 | 52.86358 | 0.568639 |
| Original median | 89.39740 | 0.377627 | 53.52948 | 0.570718 |
| Matched seed 4 | 106.80018 | 0.464118 | 62.54780 | 0.705240 |
| Matched seed 5 | 90.24736 | 0.374544 | 53.01559 | 0.573013 |
| Matched seed 6 | 88.17690 | 0.376129 | 54.35321 | 0.569624 |
| Matched median | 90.24736 | 0.376129 | 54.35321 | 0.573013 |

All 15 distinct quantized artifacts passed managed verification and
weighed exactly 346,757,888 bytes. Full CPU and CUDA gradient sweeps
selected identical exchanges in both folds. The worst per-tensor relative
gradient error was 0.000137 on Dolly and 0.000291 on W4. W4 selection
recreated the earlier X4c override map exactly; the uniform and original
controls reproduced their earlier W4 scores. The two folds reuse the
same small corpora and model. They do not establish transfer to other
models or broader text collections.

## Inputs and controls

- **Dolly:** Databricks, [databricks-dolly-15k, revision
  `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
  CC BY-SA 3.0. Use the eight A1 `held_out` instruction rows from each
  of `creative_writing` and `open_qa`, in cache order. Normalize whitespace.
  Do not include the responses or A1 system prompts. The harness reads the
  local A1 cache, checks the raw and split hashes, and checks each instruction
  against the raw data. It does not download data or publish the subset.
  The resulting 16-line text has SHA-256
  `2c8296319e4d9c2829447fde4b1edf3203d56749625c6fa6eea675ae6ab15bb5`.
- **W4:** Brewery-authored `crates/eval/data/edit-checks/heldout.txt`,
  16 lines, SHA-256
  `9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923`.
  There are no shared lines after case and whitespace normalization.
- Each fold ranks 170 tensors by absolute first-order NLL damage on its
  selection corpus. Each recipe uses three disjoint, equal-shape
  type exchanges. Use the same uniform recipe, original random seeds 1 to 6,
  and three exchange-matched random controls per fold. A matched control
  has the same shapes, type transitions, and bytes moved as the corpus
  recipe; it selects tensor names at random. Use seeds 4 to 6 first and skip
  duplicate recipes before evaluation. Report the two random groups
  separately.
- Score each fold only on the other corpus, using teacher forcing and
  KL(BF16 || recipe) over the full vocabulary. The evaluator starts with
  the first word as the prefix on each line and scores the rest. This
  gives 222 W4 and 186 Dolly evaluation positions; gradients use all
  next-token positions (222 W4 and 188 Dolly).

## Replay

Run on the qualification host. Use the X4 and X4c evidence directories
in `/arboretum` and the local A1 cache. Keep models and worker data in
the resolved `x4g` scratch directory. Do not use the live controller.

1. Under the exclusive build gate, build the Brewery controller, engine,
   and worker packages from `83abda6` with `--locked --bins`,
   `CARGO_PROFILE_DEV_DEBUG=0`, and `CARGO_INCREMENTAL=0`. Copy `brewery`,
   `brewery-dispatch`, `brewery-engine`, and `brewery-worker` into
   `x4g/bins`. The conversion stage checks their hashes against the X4c
   acceptance binaries. It also checks the quantizer, evaluator, and W0
   inference libraries.
2. Use the X4c evidence script `prepare-handoff.py` with its output
   directory set to `x4g/handoff` to recreate the expanded F32 GGUF and
   original tensor list. Check the Lanternfish seal before using the pin.
   Use NumPy and the `gguf-py` decoder in the W0 llama.cpp checkout for
   attribution.
3. From this repository, run the commands below. `plan` records both
   folds before `convert` starts. `convert` runs a fresh isolated controller
   and uses its managed profile and conversion APIs. The scratch worker
   must verify each artifact, and the harness checks its size and hash.
   The harness reuses an artifact across folds only when the entire
   override map is identical.

```sh
export TMPDIR=/arboretum/tmp OPENBLAS_NUM_THREADS=1
s=/mnt/games/model-brewery/xa-scratch/x4g
e=/arboretum/x4g-evidence
pin=/arboretum/pins/lanternfish-x4c-e0fd1b7
q=$HOME/.local/bin/resource-queue
(cd "$pin" && sha256sum -c SHA256SUMS)
python -B qualification/x4_generalization.py prepare
for corpus in dolly w4; do
  for backend in cpu cuda; do
    set -- env LD_LIBRARY_PATH="$pin/lib/$backend" "$pin/lanternfish" \
      --verbose grad-corpus --backend "$backend" --threads 12 \
      --model "$s/handoff/reference.f32.gguf" --spec "$s/handoff/$corpus.json" \
      --corpus "$s/handoff/$corpus.txt" --out "$s/$corpus-$backend"
    if [ "$backend" = cuda ]; then
      "$q" run --resource gpu --wait 600 --max-hold 120 --holder x4g-grad -- "$@" \
        > "$e/$corpus-$backend.log" 2>&1
    else
      "$@" > "$e/$corpus-$backend.log" 2>&1
    fi
  done
  python -B /arboretum/lanternfish-x4c/tools/check_corpus_parity.py \
    --cpu "$s/$corpus-cpu" --gpu "$s/$corpus-cuda" \
    --spec "$s/handoff/$corpus.json" --corpus "$s/handoff/$corpus.txt" \
    --log "$e/$corpus-cuda.log" --report "$e/$corpus-parity.json"
done
python -B qualification/x4_generalization.py plan
python -B qualification/x4_generalization.py convert
```

`plans.json` records the gradient summaries, exchanges, and complete
profile overrides. `results.json` records full-precision perplexity and
mean KL, the exact random comparisons, and links to managed verification
receipts. Read the two folds separately. This sample does not establish
transfer beyond these corpora or this model.
