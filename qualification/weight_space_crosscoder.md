# X6 crosscoder model diff acceptance

The tests use Qwen2.5-0.5B-Instruct BF16 on a private loopback controller.
They do not use the live workbench or a GPU. These results are
**not release-qualified**.

## Replay

Follow [Crosscoder model diff](../OPERATOR.md#crosscoder-model-diff) to train
and import a report. For the Qwen replay:

1. Use the [W9 CPU ROME procedure](weight_space_stage2.md) for CounterFact
   case 7. Change the answer to "Google" for "Apple A5 was created by" at
   layer 12. Use 16 optimization steps.
2. Use `prune_model` with the W12 `magnitude-30` recipe in
   `weight_space_pruning_acceptance.py` for the same parent. It applies a
   30% mask to the selected layer-8 tensors.
3. Convert the parent and both children to private BF16 GGUFs with the same
   assets. Capture pairs with one W1 pin. Use `ffn_out` at layer 12 for ROME
   and `l_out` at layer 8 for pruning.
4. Fit a 192-feature dictionary with `--k 32 --steps 240`. Use
   `--difference-weight 16` for ROME and the default of 1 for pruning.
   Use `qualification/crosscoder_evidence.py` to check REST, MCP, and
   downloaded weights. Supply the held-out prompt anchors for ROME.

The pinned test corpus has 281 training pairs of activation rows and 74
held-out pairs per model comparison. Training uses 16 pinned W4 lines in
two text blocks and a block of four alternate Apple A5 phrases. Holdout
uses the target, two paraphrases, two locality prompts, and three unrelated
prompts. The trainer checks token IDs, positions, row widths, runtime,
and model hashes before the fit. The evidence directory on the acceptance
host is `/arboretum/x6-evidence/`. It contains source pins, the trace
specifications, and the small replay harness.

## CPU results

### ROME, layer 12

The scratch weights match the W9 CPU child hash. The held-out edit prompt
ranks the replacement token first in the child. The crosscoder marks 187
directions changed and 5 shared. Neither side has a unique or dead feature.
Parent/child held-out MSE is 0.01698/0.01726, explained variance is
0.0134/0.0797, and L0 is 31.61/31.61.

Feature 2 is changed: its child norm ratio is 0.559 and its decoder cosine
is 0.672. At the subject-final token in the primary prompt and both
paraphrases, the feature's signed projection is 31.9%, 38.4%, and 39.1%
of the measured activation change, respectively. It has zero activation at
the eight recorded next-token, locality, and unrelated positions. This is
an association in these prompts, not a causal or semantic identification.

Absolute reconstruction is weak. The optional 0.50 explained variance
target did not pass on either side. A larger dictionary did not help.
With training position 0 removed, explained variance was 0.0346/0.0856
on the full holdout and 0.0137/0.0669 on the holdout without position 0.
Do not use this small corpus as proof of general reconstruction quality.

The real ROME artifact loaded in the isolated workbench at desktop and
mobile widths. REST, MCP, and the weights download agreed.

### Magnitude pruning, layer 8

The tensor diff found 1,307,443 changed elements in each of the gate, up,
and down projections, and no other changed tensors. The crosscoder marks
105 directions changed, 46 shared, and 41 dead. Neither side has a unique
feature. The child norm ratios range from 0.4952 to 0.5092.
Parent/child held-out MSE is 5.10549/5.27617, explained variance
is 0.98129/0.98067, and L0 is 31.89/31.89. Both sides have 41 dead
features. REST, MCP, and the weights download agreed.

The MSE values use the native activation units. Do not compare absolute
MSE across these two layers and roles.

### Identical-input controls

The child norm ratio is `child / (parent + child)`. A live feature is
unique to the parent at 0.10 or less, or to the child at 0.90 or more.
For other live features, decoder cosine below 0.995 means "changed."
When both inputs use the same parent activations and the same training
settings, neither layer has a changed or unique feature. The minimum live
decoder cosine rounds to 1.0 in both controls. Explained variance on
identical inputs is 0.0968 at layer 12 and 0.9873 at layer 8. These
controls test for drift on identical inputs, not the significance of
each change at the selected threshold.
