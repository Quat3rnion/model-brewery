# Stage 2: locate-then-edit on Qwen

Run the CPU acceptance harness on an isolated loopback controller:

```sh
export TMPDIR=/arboretum/tmp
export UV_CACHE_DIR=/arboretum/tmp/brewery-w6-uv-cache
uv run --no-project --offline --with mcp==1.26.0 python -B \
  qualification/weight_space_stage2.py --evidence /arboretum/w9-accept/evidence
```

Use the W8 harness pins and a Stage 2 build of the Brewery controller,
worker, dispatcher and engine. Put the Stage 2 Lanternfish executable at
`/arboretum/w9-runtime/lanternfish`; build its four libraries in
`/arboretum/llama-w9/build/bin` with patches 0001 to 0003. The Brewery engine
checks the compiled SHA-256 pins. Do not replace the W0 build or binaries.
The harness needs free space for one parent and one child SafeTensors file,
their private GGUF files, and the converter's temporary files. It discards
each child after it saves evidence and removes the isolated stack at exit.

The harness submits ROME for CounterFact case 7, then submits MEMIT and
AlphaEdit for the same five cases (7, 0, 1, 2 and 3) from the W4 pinned
subset. It uses one BF16 Qwen2.5-0.5B-Instruct parent. It checks that each
edit changes only the specified MLP down projections and keeps BF16.
It runs W4 integrity, load and generation smoke, and all 17 behavior
metrics for each child. It also uses the pinned W4 runner on only the edit
cases, paraphrases and locality prompts. A whole-subset efficacy average
is not a substitute for edit-case efficacy.

The reference equations come from [Meng et al., ROME (2022)](https://arxiv.org/abs/2202.05262),
[Meng et al., MEMIT (2023)](https://arxiv.org/abs/2210.07229), and
[Fang et al., AlphaEdit (2024)](https://arxiv.org/abs/2410.02355).
This implementation uses fixed short contexts, a small Apache-2.0 sample,
selected layers, a regularized second moment, and a first-batch AlphaEdit
projector. It does not reproduce the published benchmark configurations.

## Isolated CPU result

All three MCP jobs made reversible scratch children. Each child kept BF16
and changed only the specified down projections. All three W4 integrity
and generation checks passed. The worker recorded 17 behavior metrics
for each child. The harness did not publish any child.

| Method | Edited targets at rank 1 | Paraphrases at rank 1 | Locality probability change | Held-out perplexity change |
| --- | ---: | ---: | ---: | ---: |
| ROME | 0/1 to 1/1 | 0/2 to 0/2 | +0.824 points | 81.0388 to 81.0202 |
| MEMIT | 0/5 to 5/5 | 0/10 to 4/10 | -0.669 points | 81.0388 to 80.7887 |
| AlphaEdit | 0/5 to 5/5 | 0/10 to 4/10 | -0.606 points | 81.0388 to 80.9424 |

Locality is the change in mean probability of the original answers on
the matching neighborhood prompts, in percentage points. On common case
7, the edit target became rank 1 for all three methods. Its probability
went from 1.12% to 97.03%, 98.11%, and 98.11%, respectively. None of the
three made the target rank 1 on either paraphrase of that common case.
AlphaEdit and MEMIT had the same top-1 scores and similar locality drift
on the five-case batch. This sample does not establish a locality advantage.

Evidence: `/arboretum/w9-accept/verified/report.json` and the ten job
receipts in that directory. The harness removed the temporary model files.
Results are **not release-qualified**.
