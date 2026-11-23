---
name: brewery-weight-space
description: Run Model Brewery weight-space experiments through MCP or REST. Use for CPU traces, activation attribution, interventions, SAE features, Fisher tensor sensitivity, scratch edits, fact edits, merges, pruning, compiled programs, checks, diffs, or publish requests.
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
   with `diff_tensors`, `diff_traces`, and `diff_behavior` as applicable. If
   the bundle has `token_loss`, use `read_token_loss_diff` with the immediate
   parent `a` and child `b`. Use `sentence`, `start`, and `end` to limit the
   read. Report the signed share of net log-probability gain and the check
   bundle ID.
6. If the user asks, use `request_publish` for the exact child hash. If smoke
   or behavior did not pass, disclose that and acknowledge the current bundle
   only after review. Integrity failure blocks the request. A request leaves
   the Library unchanged. Report the IDs, measured results, and blockers as
   **not release-qualified**.

## Choose a method

| Goal | Method |
| --- | --- |
| Change known weights | `apply_edit` for slices, low-rank updates, or projection removal. See [Scratch weight revisions](../../../OPERATOR.md#scratch-weight-revisions). |
| Measure Fisher sensitivity | Use `tensor_sensitivity` with one pinned GGUF, the W4 held-out corpus SHA-256, and a small tensor or layer selection. Read `get_job` and record the revision, corpus, backend, and observation count. This job does not edit weights. See [Tensor sensitivity](../../../OPERATOR.md#tensor-sensitivity). |
| Find activation circuits | Use `find_circuit` with one pinned Library GGUF, equal-length clean and corrupt prompts, and a small site selection. Read `get_job`. Cite `/api/jobs/{id}/circuit` and distinguish signed attribution from measured patch deltas. Report Pearson `r` and Spearman `rho` only for the patched sites. See [Activation attribution](../../../OPERATOR.md#activation-attribution). |
| Change model facts | `rome_edit` for one fact at one layer; `memit_edit` for batches across layers; `alphaedit_edit` to project away from sampled preserved keys. Start with a private BF16 parent GGUF. See [Locate and edit a fact](../../../OPERATOR.md#locate-and-edit-a-fact). |
| Combine pinned revisions | `merge_models` on compatible layouts. See [Merge revisions](../../../OPERATOR.md#merge-revisions). |
| Test pruning | `prune_model` for a reversible dense mask, not a smaller model. `plan_experts` can produce a REAP map; `prune_experts` returns 503 until a private writer is available. See [MCP pruning](../../../OPERATOR.md#mcp-for-weight-space-work). |
| Compare weight spectra | Queue `weight_spectra` with a scratch or approved local Library `id`. A scratch input uses its parent unless you give `compare_id`. Start with one `layer` or exact `tensor`. Read the result with `get_job`. Record the hashes, method, and energy fraction. See [Spectral weight health](../../../OPERATOR.md#spectral-weight-health). |
| Check model lineage | Call read-only `lineage` with a Library `id`. Use `candidate_1` through `candidate_3` to select local entries. To check a merge, include the base and two possible parents. Report the fingerprint evidence and do not claim provenance. See [Model lineage](../../../OPERATOR.md#model-lineage). |
| Change activations only | Use `start_trace` with [trace interventions](../../../OPERATOR.md#trace-interventions). Reuse [versioned rules](../../../OPERATOR.md#library-intervention-artifacts). Use `read_feature_trace` to inspect [SAE features](../../../OPERATOR.md#sparse-autoencoders-saes) or `steer_feature` to apply an SAE direction. Compare to an unmodified trace; there is no weight revision to publish. |
| Build a standalone program | Use `compile_program` for a supported named program and `run_program` for examples. Run checks and read `program_correctness` and `program_cases`. Failure blocks publication. Acknowledge the bundle before requesting publication. The GGUF, trace, and weight diff steps do not apply. See [Compiled programs](../../../OPERATOR.md#compiled-programs). |

## Limits and evidence

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
