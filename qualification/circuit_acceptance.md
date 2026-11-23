# X5 CPU acceptance: factual recall

Use the pinned Qwen2.5-0.5B-Instruct BF16 GGUF. The clean text is
`The capital of France is`. The corrupt text is `The capital of Germany is`.
The tokenizer gives five IDs for each prompt:

- Clean: `[785, 6722, 315, 9625, 374]`
- Corrupt: `[785, 6722, 315, 9856, 374]`

The prompts differ at the country token only. The objective is the next-token
logit for ` Paris` (12095) minus the logit for ` Berlin` (19846). The sweep
uses layers 0, 12, and 23, all five positions, and all three roles
(`l_out`, `attn_out`, `ffn_out`). It measures 45 sites and patches the top 10
and five lower ranks with independent W2 replacement runs. The lower-rank
sample uses a fixed seed.

Copy the controller, dispatch, worker, and engine binaries from the same
build to a private directory. Use the dedicated scratch disk shown below
because each BF16 gradient can need a temporary F32 copy of the model.
Use the disk queue if you must run on the shared system disk instead.
Do not point the script at a running workbench.

```sh
export TMPDIR=/arboretum/xa-scratch/x5
mkdir -p "$TMPDIR"
python3 -B qualification/circuit_acceptance.py \
  --bins /arboretum/brewery-x5-bins \
  --pin /arboretum/pins/lanternfish-x3-fa0e8c5 \
  --evidence /arboretum/brewery-x5-evidence \
  --scratch "$TMPDIR"
```

The script checks the model and runtime pins, submits `find_circuit` to the
isolated controller, and reads the saved result through REST. It checks
that the clean and corrupt margins have opposite signs, that the top 10
and five lower-ranked sites have patch measurements, and that the
strongest patch increases the Paris margin. It calculates Pearson `r`
and Spearman `rho` again from the 15 signed pairs. `artifact.json`
records all values; `report.json` records the patch results, correlations,
and binary hashes.
These measurements are **not release-qualified**.
