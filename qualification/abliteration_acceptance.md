# Abliteration acceptance on Qwen2.5-0.5B-Instruct

Not release-qualified. No edited model was approved or published.

## Protocol

Use an isolated controller on a random loopback port, a private stack, and
the pinned Qwen2.5-0.5B-Instruct SafeTensors parent. Native fit, screen,
conversion and checks use the CPU. Heretic uses the GPU queue with one
offline `noslop-dolly` run (seed 83, two trials). Convert the byte-identical
parent copy and all children with the same tokenizer and model assets.
The stock BF16 GGUF matches A5 and A6.

The native arm uses the MCP tools `compute_direction`, `get_direction`,
`search_ablation`, `get_scratch`, and `get_checks`. Use seed 7 and the first
three A5 trials with at most one retained child. The Heretic arm uses
`run_heretic`, `get_scratch`, `convert_scratch_gguf`, `run_checks`, and
`get_checks`. The baseline and both arms share the A5 sycophancy holdout
(32 prompts), A6 slop holdout (32 prompts), and length/KL set (8 prompts).
Do not use the A5 slop set for the comparison: it overlaps Heretic's
optimization data. Compare generated flags in cache row order. Use exact
two-sided McNemar tests and Holm correction across the six slop and
sycophancy comparisons. Wilson 95% intervals describe the separate rates,
not the paired differences.

For benign refusal, replay only the 16 reviewed A1r prompts with the same
CPU runner on the exact checked GGUFs. Record flags and rates only. This
measures over-refusal, not compliance with harmful requests. It does not
produce a direction. This separate check avoids rerunning the rest of W4
and A2. Managed GSM8K requires an HGN bundle, not a private scratch GGUF.
GSM8K was not measured.

## Reproduce

Build the controller, dispatcher, engine and worker in the private A7
target. Copy them out of the target and record their hashes. The harness
checks the archived A5/A6 model and runner hashes against the local files
and compares the W4 metrics with the archived receipts. It stages a
private copy of the installed A6 runtime environment and uses the A7
launcher to collect scalar scores from the Optuna journal.

```sh
scratch=/mnt/games/model-brewery/xa-scratch/a7
TMPDIR=/arboretum/tmp PYTHONDONTWRITEBYTECODE=1 \
  python3 qualification/abliteration_acceptance.py \
  --root "$scratch/replay" --bins "$scratch/bins" \
  --runner /arboretum/a2-accept/bins/scratch-check-logits \
  --runtime /arboretum/heretic --resource-queue "$HOME/.local/bin/resource-queue" \
  --evidence /arboretum/tmp/a7-evidence
```

The CPU-only A6 smoke run used the same `noslop-dolly` configuration and
exported weights. Its CPU export hash is different from the CUDA export.
Do not use that artifact as the scored Heretic child.

## Evidence and checks

- `/arboretum/tmp/a7-evidence/`: isolated MCP receipts, binary hashes,
  checked GGUF hashes, paired flags and comparison report.
- `/arboretum/tmp/a7-shots/`: mocked workbench screenshots at 1280 and
  390 px, in both themes.
- From `web/`: `npm run check`, `npm run build`, and
  `npm run test:flows -- abliteration.spec.mjs`. The mocked controller
  provides GET receipts only. Axe checks the workbench and check panel for
  WCAG A and AA violations.
- From the repository root: `python3 -m unittest discover -s qualification
  -p test_abliteration_acceptance.py`. Run the Heretic launcher tests in
  the pinned Python environment with `python -m unittest discover -s
  tools/heretic -p test_run.py`. Run the Rust MCP, OpenAPI, hosted-agent,
  scratch and Heretic receipt tests inside the exclusive build gate.

## What the results show

The native fit selected L10 (held-out Cohen's d 24.369, margin 1.330).
The three screen trials matched A5. The trial 0 fitted direction reduced
the first-token agreement margin by 0.02140 logits against a matched
random reduction of -0.00098. It beat the control on 5/8 prompts. This
is a small change in the signed stock margin (0.48216 logits), despite
the large separation score. A3 and A5 did not show significant flips
in generated replies.

Heretic's optimization scorer found slop on 0/4 stock prompts and 0/4
prompts for each trial. Trial KL was 15.02032 and 0.04216 nats. Heretic
selected the second trial, and the imported SafeTensors and private BF16
GGUF matched the A6 CUDA run. These trial metrics are not the held-out
A2 measurements. The A6 held-out run found stock slop on only 2/32
prompts, and did not establish a reliable reduction.

The acceptance stack based on `cc1b124` ran the paired checks. A separate
GPU-queued MCP import used the updated launcher and validated the persisted
trial journal. It produced the exact same SafeTensors SHA-256. See
`final-journal/child.json` and `final-bin.sha256` in the evidence directory.
No CPU checks were repeated for an identical Heretic export.

| CPU check | Stock | Native trial 0 | Heretic trial 2 |
| --- | ---: | ---: | ---: |
| Sycophancy agreement | 27/32 | 26/32 | 27/32 |
| Slop markers | 2/32 | 2/32 | 0/32 |
| Benign refusal | 0/16 | 0/16 | 0/16 |
| First-token KL (nats) | 0 | 0.004437 | 0.182659 |
| Continuation KL (nats) | 0 | 0.000874 | 0.020238 |
| W4 perplexity | 81.03884 | 80.43740 | 83.89473 |
| CounterFact locality probability | 0.049344 | 0.050396 | 0.050770 |
| zsRE locality probability | 0.275715 | 0.275699 | 0.280801 |

Native agreement flags changed on 1/32 prompts (one loss and no gains).
Heretic changed six agreement flags (three losses and three gains). Both
comparisons with stock had McNemar p = 1.0. The absence of benign refusals
on 16 prompts gives a Wilson 95% interval of 0%-19.36% for each model.

Native changed no slop flags. Heretic changed two from hit to no hit
(exact McNemar p = 0.5). Stock's slop rate has a Wilson 95% interval of
1.73%-20.15%; Heretic's interval is 0%-10.72%. All six slop and sycophancy
comparisons have Holm-adjusted p = 1.0. The data do not show a measurable
reduction for either child. Both completed W4 integrity, smoke and behavior
checks. Heretic changed the first-token distribution more and had higher
perplexity than stock.
