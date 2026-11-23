---
name: brewery-weight-space
description: Run Model Brewery weight-space experiments through MCP or REST. Use for CPU traces, interventions, behavior directions, abliteration, Heretic, SAE features, scratch edits, fact edits, merges, pruning, compiled programs, checks, diffs, or publish requests.
---

# Weight-space experiments

Connect a Streamable HTTP MCP client to the trusted workbench at
`http://127.0.0.1:4141/mcp` or its approved HTTPS URL with `/mcp`. Get the
schemas with MCP `tools/list`. For REST, use `tools/brewery-agent describe`,
`state`, `create`, and `wait`. Use the [OpenAPI contract](../../../api/openapi.json)
for routes that the CLI does not wrap. See
[MCP access](../../../OPERATOR.md#mcp-for-weight-space-work) and
[hosted agent setup](../../../OPERATOR.md#hosted-workbench-agent).
If you copied this skill, open its linked files in the Model Brewery checkout.

Change weights only in scratch. Keep the original files and existing Library
entries unchanged. Approval and deployment are operator-only, also when using
REST. Keep secrets out of goals, tool arguments, and outputs.

## Scratch loop

1. Find the input with `list_models` or add it with `register_model`. Confirm
   the worker capability and pin the input ID and SHA-256.
2. Capture the baseline with `start_trace` and read it with `read_trace`.
   For a SafeTensors input, use `create_scratch` and `convert_scratch_gguf`
   first. Save the trace ID, prompts, and capture settings.
3. Use `list_scratch` to find the pinned scratch parent. Select a method below
   and record the child ID and hash. For queued work, use `get_job` to read
   the terminal result before starting the next step.
4. Use `convert_scratch_gguf` on the child with the same assets as the parent.
   Trace the child with the same prompts, positions, and capture settings.
5. Use `run_checks`, wait for any check job, then read `get_checks`. Compare
   with `diff_tensors`, `diff_traces`, and `diff_behavior` as applicable.
   To score a behavior direction, pass `behavior_direction.prompt_sets` with
   the verified held-out set digests. Use the same digests for each run.
   Add the approved `benign_refusal` digest to measure over-refusal on 16
   reviewed Databricks Dolly 15K prompts (CC BY-SA 3.0). Report flags and
   `benign_refusal_rate`, not replies or a refusal direction. This does not
   measure harmful-set compliance. Get operator approval before you publish
   these research results.
   See [Behavior direction checks](../../../OPERATOR.md#behavior-direction-checks).
6. If the user asks, use `request_publish` for the exact child hash. If smoke
   or behavior did not pass, disclose that and acknowledge the current bundle
   only after review. Integrity failure blocks the request. A request leaves
   the Library unchanged. Report the IDs, measured results, and blockers as
   **not release-qualified**.

## Choose a method

| Goal | Method |
| --- | --- |
| Change known weights | `apply_edit` for slices, low-rank updates, or projection removal. See [Scratch weight revisions](../../../OPERATOR.md#scratch-weight-revisions). |
| Change model facts | `rome_edit` for one fact at one layer; `memit_edit` for batches across layers; `alphaedit_edit` to project away from sampled preserved keys. Start with a private BF16 parent GGUF. See [Locate and edit a fact](../../../OPERATOR.md#locate-and-edit-a-fact). |
| Combine pinned revisions | `merge_models` on compatible layouts. See [Merge revisions](../../../OPERATOR.md#merge-revisions). |
| Test pruning | `prune_model` for a reversible dense mask, not a smaller model. `plan_experts` can produce a REAP map; `prune_experts` returns 503 until a private writer is available. See [MCP pruning](../../../OPERATOR.md#mcp-for-weight-space-work). |
| Change activations only | Use `start_trace` with [trace interventions](../../../OPERATOR.md#trace-interventions). Reuse [versioned rules](../../../OPERATOR.md#library-intervention-artifacts). Use `read_feature_trace` to inspect [SAE features](../../../OPERATOR.md#sparse-autoencoders-saes) or `steer_feature` to apply an SAE direction. Compare to an unmodified trace; there is no weight revision to publish. |
| Remove a behavior direction during inference | Use `get_direction`, then `create_direction_intervention` with `project_out` and the target rows. Pass the saved version to `start_trace` and compare the result with `diff_traces`. Use `subtract` to export a fixed scaled direction. `export_control_vector` cannot export projection. |
| Measure a behavior direction | Use `compute_direction` with a pinned Library Qwen2 GGUF, behavior, and increasing layer numbers. Read version 2 with `get_direction`. Report held-out Cohen's d, not a change in generated behavior. See [Behavior directions](../../../OPERATOR.md#behavior-directions). |
| Ablate a fitted direction | Use `ablate_direction` with a pinned scratch parent and a version 2 direction. Convert the private child and compare behavior and locality against its parent. See [Native directional ablation](../../../OPERATOR.md#native-directional-ablation). |
| Search ablation settings | Use `search_ablation` when a slop or sycophancy direction is fitted on the exact parent GGUF. Set the trial, child, and disk limits. Read `get_job` for the matched control, Pareto front, and paired checks. A screen effect is not a weight-edit result. See [Native ablation search](../../../OPERATOR.md#native-ablation-search). |
| Run or reproduce a Heretic edit | Use `run_heretic` with a standalone Library SafeTensors ID and an offline operator profile. Use `reproduce_heretic` only with a v4 manifest and a verified matching Hugging Face download. Use a byte-identical scratch copy for the parent's private GGUF, then convert and check the child with the same assets. See [Heretic jobs](../../../OPERATOR.md#heretic-jobs). |
| Build a standalone program | Use `compile_program` for a supported named program and `run_program` for examples. Run checks and read `program_correctness` and `program_cases`. Failure blocks publication. Acknowledge the bundle before requesting publication. The GGUF, trace, and weight diff steps do not apply. See [Compiled programs](../../../OPERATOR.md#compiled-programs). |

## Limits and evidence

- For an ablation comparison, follow the [MCP abliteration workflow](../../../OPERATOR.md#abliteration-workflow-through-mcp).
  Fit the native direction on the exact baseline GGUF. Give Heretic the
  original Library SafeTensors ID. Report fitted probe scores separately
  from paired child results. Use identical, disjoint held-out
  `run_checks` pins for the children and record the hashes, flip counts,
  intervals, KL, locality and perplexity.
- On Qwen2.5-0.5B-Instruct, the tested native directions carried only a
  small part of the sycophancy signal. A3 and A5 found no significant paired
  flips. Stock slop was too rare to measure a reliable reduction. Heretic
  `noslop` did not show a measurable reduction either.
- Use single-file SafeTensors on an enabled local editor. For direct edits,
  dense pruning, and fact edits, the 32 MiB guard counts the full size of all
  affected tensors, even for a slice. Fact edits support bounded dense Llama
  and Qwen2. Use `backend: cpu` by default. Use `backend: cuda` or
  `backend: rocm` only with the matching pinned runtime and operator GPU
  admission. If no GPU is visible, report the job failure. See [Locate and
  edit a fact](../../../OPERATOR.md#locate-and-edit-a-fact).
- Use narrow trace captures. The hard trace cap is 32 MiB. Read traces
  before the worker removes old results. A successful job can return 404
  for expired trace data. Check [CPU trace
  caps and retention](../../../OPERATOR.md#cpu-trace-jobs) and the MCP byte
  limits before submitting or reading a large capture.
- Set an idempotency key before a write. Use the same key and arguments for a
  retry. Where the job schema supports it, use `allow_duplicate: true` and a
  new key only for a deliberate repeat.
- Read [the check bundle](../../../OPERATOR.md#checks-before-a-publish-request)
  for the exact child hash. Integrity-only is not a measured behavior pass.
  Report target, paraphrase, locality, and perplexity separately. Use the
  immediate parent as `a` and child as `b` for the behavior diff. Read each
  metric's direction and threshold; a missing value is not a pass. In
  [diff reports](../../../OPERATOR.md#compare-revisions-and-traces), tensor
  rank is sampled and logit KL covers only shared top-k tokens.
