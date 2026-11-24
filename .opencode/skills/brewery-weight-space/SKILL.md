---
name: brewery-weight-space
description: Run Model Brewery weight-space experiments through MCP or REST. Use for CPU traces, activation attribution, interventions, behavior directions, abliteration, Heretic, SAE features, crosscoders, Fisher tensor sensitivity, scratch edits, fact edits, merges, pruning, compiled programs, interpretability benchmark scores, checks, diffs, findings, replay, or publish requests.
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
2. Record the hypothesis with `create_finding` before running an experiment.
   Pin the model revision ID and SHA-256. Then capture the baseline with
   `start_trace` and read it with `read_trace`.
   For a SafeTensors input, use `create_scratch` and `convert_scratch_gguf`
   first. Save the trace ID, prompts, and capture settings.
3. Use `list_scratch` to find the pinned scratch parent and `get_scratch` to
   read its provenance. Select a method below
   and record the child ID and hash. For queued work, use `get_job` to read
   the terminal result before starting the next step.
4. For an HGN child, skip GGUF conversion. Call `run_checks` with the child
   ID and exact hash. Read `hgn_stored` in `get_checks` for the decoded-store
   audit and CPU forward scores. Require both checks. Report scores and flags,
   not generated text. See [stored HGN checks](../../../OPERATOR.md#native-hgn-v2-scratch-edits).
   For SafeTensors, use `convert_scratch_gguf` on the child with the same assets as the parent.
   Trace the child with the same prompts, positions, and capture settings.
5. Use `run_checks`, wait for any check job, then read `get_checks`. Compare
   with `diff_tensors`, `diff_traces`, and `diff_behavior` as applicable. If
   the bundle has `token_loss`, use `read_token_loss_diff` with the immediate
   parent `a` and child `b`. Use `sentence`, `start`, and `end` to limit the
   read. Report the signed share of net log-probability gain and the check
   bundle ID.
   To score a behavior direction, pass `behavior_direction.prompt_sets` with
   the verified held-out set digests. Use the same digests for each run.
   Add the approved `benign_refusal` digest to measure over-refusal on 16
   reviewed Databricks Dolly 15K prompts (CC BY-SA 3.0). Report flags and
   `benign_refusal_rate`, not replies or a refusal direction. This does not
   measure harmful-set compliance. Get operator approval before you publish
   these research results.
   See [Behavior direction checks](../../../OPERATOR.md#behavior-direction-checks).
   For private GGUF, call `list_evaluation_sets`. Pass
   `registered_marker: {"name":"<set-name>","partition":"tune"}` to `run_checks`.
   Reserve `final_gate` for the final check after you fix the candidate.
   Neither partition is available to direction fitting or ablation search.
   In `get_checks`, report marker rates, per-class counts and flags for both
   sides. Do not repeat reply prefixes.
   See [Registered evaluation sets](../../../OPERATOR.md#registered-evaluation-sets).
6. Use `add_evidence` to select numeric paths in the finished artifacts and
   record a verdict with your reasoning. Use `get_finding` to verify the
   saved version. Use `list_findings` to find earlier findings. Use
   `replay_finding` to replay an
   experiment on another pinned revision; see [Findings](../../../OPERATOR.md#findings). Compare the same paths and check
   the runtime pins before giving the replay a verdict.
7. If the user asks, use `request_publish` for the exact child hash. If smoke
   or behavior did not pass, disclose that and acknowledge the current bundle
   only after review. Integrity failure blocks the request. A request leaves
   the Library unchanged. Report the IDs, measured results, and blockers as
   **not release-qualified**. Use `discard_scratch` to remove a child that
   you do not need.

## Choose a method

| Goal | Method |
| --- | --- |
| Change known weights | `apply_edit` for SafeTensors slices, low-rank updates, or projection removal. See [Scratch weight revisions](../../../OPERATOR.md#scratch-weight-revisions). |
| Measure Fisher sensitivity | Use `tensor_sensitivity` with one pinned GGUF, the W4 held-out corpus SHA-256, and a small tensor or layer selection. Read `get_job` and record the revision, corpus, backend, and observation count. This job does not edit weights. See [Tensor sensitivity](../../../OPERATOR.md#tensor-sensitivity). |
| Find activation circuits | Use `find_circuit` with one pinned Library or private scratch GGUF, equal-length clean and corrupt prompts, and a small site selection. Read `get_job`. Cite `/api/jobs/{id}/circuit` and distinguish signed attribution from measured patch deltas. Report Pearson `r` and Spearman `rho` only for the patched sites. See [Activation attribution](../../../OPERATOR.md#activation-attribution). |
| Map quantization damage | Use `quantization_damage_map` with two pinned GGUFs from the same revision. Select `target_logit` for logit drift and measured swaps or `corpus_nll` for mean next-token loss over all W4 held-out lines. Read `read_quantization_damage`. Report the recipe as a proposal, not an applied edit. See [Quantization damage map](../../../OPERATOR.md#quantization-damage-map). |
| Change model facts | `rome_edit` for one fact at one layer; `memit_edit` for batches across layers; `alphaedit_edit` to project away from sampled preserved keys. Start with a private BF16 parent GGUF. See [Locate and edit a fact](../../../OPERATOR.md#locate-and-edit-a-fact). |
| Combine pinned revisions | `merge_models` on compatible layouts. See [Merge revisions](../../../OPERATOR.md#merge-revisions). |
| Test pruning | `prune_model` for a reversible dense mask, not a smaller model. `plan_experts` can produce a REAP map; `prune_experts` returns 503 until a private writer is available. See [MCP pruning](../../../OPERATOR.md#mcp-for-weight-space-work). |
| Compare weight spectra | Queue `weight_spectra` with a scratch or approved local Library `id`. A scratch input uses its parent unless you give `compare_id`. Start with one `layer` or exact `tensor`. Read the result with `get_job`. Record the hashes, method, and energy fraction. See [Spectral weight health](../../../OPERATOR.md#spectral-weight-health). |
| Check model lineage | Call read-only `lineage` with a Library `id`. Use `candidate_1` through `candidate_3` to select local entries. To check a merge, include the base and two possible parents. Report the fingerprint evidence and do not claim provenance. See [Model lineage](../../../OPERATOR.md#model-lineage). |
| Read MoE routing | Capture `ffn_moe_logits`, `ffn_moe_topk`, and `ffn_moe_route_weights` on Qwen2MoE or Qwen3MoE. Use `routing_atlas` for one layer. For an expert ablation, capture `l_out` in both traces and supply the `zero_expert` job as `ablation_job_id`. Pass optional `routing_evidence` to `plan_experts`; keep the independent REAP inputs. |
| Change activations only | Use `start_trace` with [trace interventions](../../../OPERATOR.md#trace-interventions). Reuse [versioned rules](../../../OPERATOR.md#library-intervention-artifacts) with `list_interventions` and `get_intervention`, or save them with `create_intervention` and `new_intervention_version`. Use `read_feature_trace` to inspect [SAE features](../../../OPERATOR.md#sparse-autoencoders-saes) or `steer_feature` to apply an SAE direction. Compare to an unmodified trace; there is no weight revision to publish. |
| Compare learned features across revisions | Use `crosscoder_diff` with a saved ID. To build an artifact, use the offline [crosscoder procedure](../../../OPERATOR.md#crosscoder-model-diff) with unmodified paired traces and separate train and holdout prompts. Check reconstruction for both revisions before interpreting unique or changed features. |
| Remove a behavior direction during inference | Use `get_direction`, then `create_direction_intervention` with `project_out` and the target rows. Pass the saved version to `start_trace` and compare the result with `diff_traces`. Use `subtract` to export a fixed scaled direction. `export_control_vector` cannot export projection. |
| Measure a behavior direction | Use `compute_direction` with a pinned Library Qwen2 GGUF or supported HGN v2, a reviewed behavior, and increasing layer numbers. Use `last_k` to fit the mean of the last K positions. Use `l_out` on HGN. Read version 2 with `get_direction`; pass `reference_id` and `reference_layer` for HGN cosine. Report held-out Cohen's d as a probe score. See [Behavior directions](../../../OPERATOR.md#behavior-directions). |
| Use an external direction | Ask the operator to [import the vector and manifest](../../../OPERATOR.md#import-an-external-direction). Find the ID with `list_directions`. Read the manifest with `get_direction`. Compare cosine only with a version 2 direction for the same hash. Use `plain` for the edit; imports have no negative-class mean or fit scores. Read `direction_import` in `get_checks`. Import is not an MCP tool. |
| Ablate a direction | Use `ablate_direction` with the parent hash and a version 2 direction. For HGN, start from a Library or HGN scratch ID. Use a direction for its pinned Library root and select `plain`, `flat`, and alpha 1. Poll `get_job` and read native-store metrics in `get_scratch`. HGN stays private. Convert a SafeTensors child to private GGUF before comparing behavior. See [Native directional ablation](../../../OPERATOR.md#native-directional-ablation). |
| Search ablation settings | Use `search_ablation` with a direction fitted on the exact checkpoint. Use `alphas` in (0,4] and optional `last_k_direction_id` to compare a last-K fit with the single-position fit. For GGUF, use reviewed slop or sycophancy and set trial, child, and disk limits. For Library HGN v2, use the version 2 direction ID, source layer, seed, and targets. Add `second_source_layer` for rank two. Read `get_job` for matched random scores and the Pareto front. HGN makes no scratch children and stores no completions. Its response-length marker is a first-token continuation proxy. See [Native ablation search](../../../OPERATOR.md#native-ablation-search). |
| Run or reproduce a Heretic edit | Use `run_heretic` with a standalone Library SafeTensors ID and an offline operator profile. Use `reproduce_heretic` only with a v4 manifest and a verified matching Hugging Face download. Use a byte-identical scratch copy for the parent's private GGUF, then convert and check the child with the same assets. See [Heretic jobs](../../../OPERATOR.md#heretic-jobs). |
| Build a standalone program | Use `compile_program` for a supported named program and `run_program` for examples. Run checks and read `program_correctness` and `program_cases`. Failure blocks publication. Acknowledge the bundle before requesting publication. The GGUF, trace, and weight diff steps do not apply. See [Compiled programs](../../../OPERATOR.md#compiled-programs). |
| Read scores on a known circuit | Call `interp_benchmark` with no arguments. Distinguish linear attribution from exact activation patching. Recall-only rows have no negative sites and cannot measure false positives. These scores do not qualify a release. See [Compiled-program benchmark](../../../OPERATOR.md#compiled-program-benchmark). |

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
- For a supported Library `hgn` or `hgn-v2` checkpoint, use `start_trace`
  with the ID on its non-Hy4 trace worker. Use at most 2048 raw prompt tokens.
  `l_out` contains all four
  streams; `final_norm` is available only at the last layer. Use
  `options.hgn_projection` with a version 2 direction ID, source layer, and
  `residual`, `dense`, `experts`, or `all`. Add `second_source_layer` for
  rank two. Report the signed cosine before orthogonalization. Omit both
  layer endpoints for all layers. Use `options.gguf_projection` with the A5
  pin for rank one on GGUF. Both selectors accept `alpha` in (0,4] and a
  verified fit made with `last_k`. Do not combine these selectors with
  other interventions or SAE.
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
