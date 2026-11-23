# Weight-space programming: Stage 1 acceptance

The acceptance run uses Qwen2.5-0.5B-Instruct on a local CPU worker.
It starts a new controller on a loopback port and creates a new data
directory under `/arboretum/w8-accept`. It does not start a Lanternfish
server or use a shared controller. These results are **not
release-qualified**.

## Run

Build the controller, engine, worker, dispatcher, and workbench from this
checkout. Use the local build gate and an installed `web/node_modules`.
The defaults in `weight_space_acceptance.py` refer to the pinned models and
runtime files used for W8, including the offline CounterFact and zsRE
cache. Use Python 3.12 or later and a fresh `--root` for a repeat run. The
runner rejects a root outside `/arboretum`.

```sh
export TMPDIR=/arboretum/tmp
export CARGO_TARGET_DIR=/arboretum/brewery-w3-target
~/.local/bin/resource-queue run --resource build --holder w8-accept \
  --max-hold 300 --wait 120 -- bash -ec '
    cargo build --locked -p brewery-controller -p brewery-engine -p brewery-worker
    npm --prefix web run build -- --outDir /arboretum/w8-accept/web
'
python3 qualification/weight_space_acceptance.py --dry-run
export UV_CACHE_DIR=/arboretum/tmp/brewery-w6-uv-cache
export PLAYWRIGHT_MODULE=/arboretum/pw-activation/node_modules/playwright/index.mjs
export CHROMIUM_BIN=/arboretum/pw-activation/browsers/chromium-1248/chrome-linux64/chrome
uv run --no-project --offline --with mcp==1.26.0 \
  python qualification/weight_space_acceptance.py
```

Playwright enables the editor in the isolated Settings view. Next the
Python MCP client registers the inputs, runs the traces, saves and applies
a steering artifact, makes one direct edit, converts both revisions to
private BF16 GGUF, and compares tensors, traces, and behavior. The paired
trace and behavior checks use the two private conversions so that the
comparison does not include differences between converters.

The edit adds a vector to the final normalization weights. It uses the
target token ` Google` (CounterFact case 7, `Apple A5 was created by`) and
the top token from the baseline trace to select a direction. The goal is a
higher target logit, not a true fact. The trace gives the selected case's
raw logit. The behavior checks report probabilities for the pinned
target, paraphrase, and locality groups, not for each case. The worker's
CounterFact target group includes case 7.

The client-side cross-check applies the last normalization and output
projection to the captured residual. It checks the logits against the
trace before it calculates the full-vocabulary probability. This is not
a worker check. An earlier attempt to edit one embedding row was refused:
the tensor is larger than the 32 MiB scratch guard. The guard counts the
full tensor even for a slice edit and would also block a separate
`lm_head` of that size.

The workbench stays open during the run. The script confirms that MCP
cannot approve the publish request. Playwright then approves the exact
hash in the Edit sessions view and confirms the Library entry. The MCP
client verifies it is the approved SafeTensors file, not a private GGUF.
In the recorded run, a browser selector matched both the check list and
the closed dialog. After narrowing the selector, approval resumed on
the same isolated data. No jobs or checks were rerun.

GSM8K is not run. The current harness evaluates the full pinned split
and needs an HGN GPU candidate. It does not accept a private BF16 GGUF
or support a CPU subset. No deployment or release checks run.

## Evidence

The runner puts the MCP receipts, worker check report, diffs, and
screenshots in `/arboretum/w8-accept/evidence`. `report.json` summarizes
the results and records the artifact hashes. The HF snapshot has a
SHA-256 file manifest but no recorded upstream revision.

The private BF16 pair raised the case 7 target logit from 10.57295 to
14.58012 (+4.00717). The client-side cross-check estimated target
probability at 1.13% before and 23.96% after, with rank 11 before and 1
after. The saved intervention changed a shared next-token logit by up
to 4.76154. The tensor diff found 795 changed BF16 elements in one
tensor. The paired trace diff has 24 activation rows, 12 logit-lens
rows, and six model-logit rows. The final normalization edit did not
change the captured activation rows.

The worker passed integrity, load-and-generate smoke, and behavior
collection and returned 17 metrics. Passing collection does not mean
all metrics improved.

| Worker metric | Parent | Child |
|---|---:|---:|
| CounterFact target probability | 2.44% | 2.79% |
| CounterFact paraphrase probability | 1.23% | 1.15% |
| CounterFact locality probability | 4.93% | 2.88% |
| zsRE target probability | 25.84% | 24.86% |
| zsRE paraphrase probability | 25.04% | 24.14% |
| zsRE locality probability | 27.57% | 27.87% |
| Held-out perplexity | 81.04 | 83.63 |

The Trace list labels private scratch inputs `Removed entry` because
they are not in the Library. The traces open and display their results.
Use their job IDs to match them to the scratch revisions in the report.

## Stage 5: pruning

Use the same binaries and pinned CPU runtime as above. Give the script a
new root under `/arboretum`:

```sh
TMPDIR=/arboretum/tmp python3 -B qualification/weight_space_pruning_acceptance.py \
  --root /arboretum/w12-accept
```

The script uses the same held-out corpus for all dense cases. It applies
10% and 30% unstructured masks to each of the three MLP matrices in
Qwen2.5-0.5B-Instruct layer 8. It also runs separate layer 8 head,
neuron, and Wanda-style neuron masks. It records W4 checks, the W5 tensor,
trace, and behavior diffs, and the held-out perplexity curve. It deletes
the private scratch copies after use. GSM8K is not run.

All five cases passed integrity, generation smoke, and behavior
collection. Each behavior diff contains 17 metrics. The evidence is in
`/arboretum/w12-w4-w5-evidence/report.json`.

| Layer 8 only | Changed tensors | Changed bytes | Held-out perplexity |
|---|---:|---:|---:|
| None (parent) | 0 | 0 | 81.04 |
| Unstructured, 10% of weights in each of three MLP matrices | 3 | 2,607,985 | 81.19 |
| Unstructured, 30% of weights in each of three MLP matrices | 3 | 7,835,105 | 77.74 |
| Magnitude, 30% of MLP neurons | 1 | 2,608,903 | 83.52 |
| Magnitude, 4 of 14 query heads | 1 | 457,812 | 81.61 |
| Wanda-style, 30% of MLP neurons | 1 | 2,608,939 | 81.12 |

No case prunes another layer. The percentages do not measure model-wide
sparsity. The full source file has 988,097,824 bytes (about 942 MiB).
The test corpus has 16 short sentences and 222 scored tokens. A repeat
of the 30% case produced different parent and child GGUF hashes that
matched the original run. A separate check verified exactly 1,307,443
zeroed weights per edited matrix and no other changed tensor payloads.
The repeat reproduced the 4.07% decrease in perplexity. One sentence
contributed about 53% of the net increase in log probability. This small
corpus does not show that quality improved on other workloads. See
`magnitude-30-repeat.json` in the evidence directory.

The masks keep tensor shapes and configuration unchanged. Physical dense
pruning needs a scratch artifact format that can pin the updated
configuration and reverse or compare changes of shape. W3 uses a
same-size XOR delta for one SafeTensors file.

The tiny Qwen2MoE fixture uses synthetic observations to exercise REAP
planning, not quality. The MCP plan keeps three of four experts at each
routed layer. Apply returns 503 and does not create any artifact. The
managed expert workflow in review can preview or reorder GGUF, but its
apply adds a Library revision. It cannot be used for scratch. The offline
GGUF writer can prune and verify, but does not produce scratch
SafeTensors. The offline SafeTensors writer needs a separate configuration
file and does not have an admitted private scratch verification path.
Before MoE acceptance can run, a verified writer must emit a private
scratch SafeTensors child and its pinned configuration, or the plan must
approve a format-aware private GGUF child. The new child must pass
integrity, generation, quality checks, and diffs without Library
registration. These results are not release-qualified.

## Optional behavior scores

Use `qualification/behavior_scoring_acceptance.py` to check the pinned Qwen
parent twice. Pass the isolated `--root`, the `--bins` directory with copies
of the controller and worker binaries, and the `--runner` built from
`tools/scratch_check_logits.cpp`. Pass the approved normalized prompt cache
as `--contrast-cache` and its tab-separated behavior and digest index as
`--contrast-index`. Include the reviewed `benign_refusal` set in that index.
The script checks the 16 flags, the aggregate over-refusal rate, and that the
report has no replies from that set. It writes `evidence/report.json` only
when two distinct CPU worker jobs return the same scores and W5 diff. The
evidence omits all completion text. The result is not release-qualified.
Set `--job-timeout` in seconds if the CPU checks need more time.
