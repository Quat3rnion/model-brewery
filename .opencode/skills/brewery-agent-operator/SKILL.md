---
name: brewery-agent-operator
description: Operate a running Model Brewery controller without a browser. Use to discover hosts and capabilities, suggest worker paths, register models and prompts, run a one-submit model conversion, queue any job kind (including transfer preview and benchmark), follow progress with SSE, wait for a result, or cancel.
---

# Operate the controller

The controller listens on loopback (default `http://127.0.0.1:4141`). Use
`tools/brewery-agent` (Python 3, JSON/JSONL output) or `curl` from the controller.
For a fresh installation, follow `INSTALL.md`. Leave remote hosts unconfigured
unless the operator requests them. Keep credentials and GPU admission under
operator control.

Contract: `tools/brewery-agent describe` (same as `GET /api/openapi.json`;
source `api/openapi.json`). Help: `tools/brewery-agent COMMAND --help`.
Job kinds, inputs, and options: `OPERATOR.md`, "Job rules".

## Loop

Finish each step before the next.

1. **Discover.** `tools/brewery-agent describe`, then `state`. Read
   `hosts[].capabilities` and `reasons`. A kind whose capability is `false`
   returns `409` with the reason. Stop and report; do not work around it.
2. **Snapshot.** In the `state` output, save `event_cursor` with the models,
   prompts, and jobs you read. Everything after the snapshot arrives as events.
3. **Suggest paths.** `tools/brewery-agent paths HOST /abs/prefix/`. The worker
   lists existing files and directories only inside its configured
   suggestion roots (at most 32 items; `truncated` tells if more exist; an
   empty list is a valid answer). It reads no file contents and starts no
   job. Use it to find model paths.
4. **Register inputs.** Create models and prompts (step 5 form). A model needs
   `name`, `host`, absolute `path`, and `kind`. Registering does not hash or
   check the file; the worker checks when a job uses it.
5. **Submit with a key.** Choose a new `Idempotency-Key` per intended
   creation before you send it. Allowed: `[A-Za-z0-9._-]{1,128}`. Save the
   key beside the request JSON.

   ```sh
   tools/brewery-agent create jobs --key NAME.v1 --file job.json
   ```

   - `201` new object. `200 {id}` replay of the same key and same body.
   - `409` same key, different body. `400` bad or duplicate key header.
   - If the response is lost, send the same request with the same key. Never
     use a new key to retry; that queues a second job.
   - Routes: `models`, `prompts`, `jobs`. Keys are scoped per route.
6. **Follow.** `tools/brewery-agent events --after "$CURSOR"` prints one JSON
   line per change (`job.created`, `job.updated`, `model.created`,
   `model.deleted`, `prompt.created`; job events carry `state`). Events say
   what changed, not the data: use `progress ID` for a compact phase sample
   or `job ID` for the full record and results. A hash-pass percentage is
   not a whole-job percentage. The optional legacy sample names its source.
   Save the last **processed** event ID and resume with it after a
   disconnect (exit 1). On `resync` (exit 3) the cursor is stale or ahead:
   take a fresh `state` snapshot and reopen from its cursor. `--limit N`
   exits 0 after N changes. Raw SSE: `GET /api/events?after=N`, or
   `Last-Event-ID` (the header wins); comment lines are heartbeats.
7. **Result.** `tools/brewery-agent wait ID --timeout SECONDS` polls until
   terminal and prints the job detail. Exit 0 `succeeded`, 4 `failed` or
   `cancelled`, 5 timeout. A timeout does not cancel the job. Read `result`
   and `error` from the detail; artifact paths are worker-side references.
8. **Cancel.** `tools/brewery-agent cancel ID` only when the user asks or the
   job is yours and wrong. A queued job ends at once. A running job gets a
   request; poll until `state` is `cancelled` or another terminal state. A
   timeout or lost link is not proof that a job stopped.

States: `queued`, `running`, `succeeded`, `failed`, `cancelled`. After a
controller restart the same job ID is checked on its assigned worker; do not
submit a replacement.

## Workflows

Queue only what the user requested. Each stage reports separately; a passed
file check is not generation or benchmark quality.

| Goal | Job kind | Notes |
| --- | --- | --- |
| Move a bundle | `transfer_preview`, then `transfer` | Preview first: it shows missing bytes and free space. A preview is not transfer evidence. |
| Trace two GGUFs | `compare` | Two `gguf` inputs, a prompt. |
| Convert (default) | One conversion workflow | See "Convert a model" below. Do not queue the four stages by hand. |
| Diagnose one stage | `inspect`, `plan`, `convert` or `verify` | Only when the user asks for one stage. `convert` writes a new directory. `host:"auto"` is for `convert` only. |
| Generate / GSM8K | `generate`, `baseline`, `evaluate` | Need the operator opt-in on the host. Development subsets never count as release evidence. |
| Endpoint benchmark | `benchmark` | `options`: `benchmark_target` and 1 to 8 `benchmarks` IDs from the host's `benchmark_targets` and `benchmarks`. Results: skill `brewery-calibration-benchmarks`. |
| Clean cache | `cache_cleanup` | Explicit; target `host`, no inputs. |

## Convert a model

One submit runs `inspect`, `plan`, `convert` and `verify` on one worker. The
controller stores the workflow, so it survives a client disconnect and a
controller restart.

1. Register the source (and auxiliary inputs) as models. Check
   the host `capabilities` and `reasons` first; a `409` has the reason.
2. Write the request JSON: `input_ids`, optional `options` (as for
   `convert`), optional `decode_backend` (`cpu`, `hip` or `cuda`), optional
   `host`. No `host`: the source host. `"auto"`: an eligible pool worker. A
   worker ID: that worker.
3. Submit once with a new key. Save the key beside the JSON.

   ```sh
   tools/brewery-agent create conversions --key NAME.v1 --file conversion.json
   ```

   `201` is a new workflow. `200` is a replay with the same key and body; it
   returns the current workflow. `409` is a different body for the key, or a
   disabled workflow or no eligible worker. Do not use a new key to retry.
4. Poll. `tools/brewery-agent conversion ID` shows `state` (`running`,
   `succeeded`, `failed`, `cancelled`), `stage`, `job_ids`, `error`, and
   `artifact_model_id`. `running` includes a queued child job.
   `tools/brewery-agent conversions` lists all workflows. Read each stage with
   `job` or `progress` and its ID from `job_ids`. The workflow adds no event
   kind; the child jobs emit the normal job events.
5. Stop on failure. The first failed or cancelled child ends the workflow and
   no later stage starts. Report the `error` and the failed job ID. Do not
   start the later stages by hand. Submit a new workflow, with a new key, only
   if the user asks for a retry.
6. Cancel. `tools/brewery-agent cancel-conversion ID` only when the user
   asks or the workflow is yours and wrong. It cancels the active child. Poll
   until `state` is terminal. A timeout is not proof that the worker stopped.

### Components

Default: send only the source. An HF source supplies its own vision, engram
(`ngram`), MTP and tokenizer files. Do not add component inputs unless the
user asks. Do not claim that a checkpoint has a component; read
`result.components` (`source` is `input`, `embedded`, `none` or `unknown`; an
older result has no field, which means "not reported").

To replace a component of an HF source:

1. Check that the host reports `capabilities.convert_components: true` in
   `state`. If it is `false`, stop and report the `reasons` entry. Do not
   reroute to another host without the user's approval. If `component_cpu_only`
   is `true`, use only the `cpu` backend, including `decode_backend`.
2. Register the component as a model with the kind `assets`, `vision`,
   `ngram` or `mtp`. Give the host and an absolute path on that host.
   Registration is not an upload and checks no bytes. Transfer first if the
   file is on another host.
3. Formats. `assets`: a flat directory of tokenizer and config files.
   `vision`, `ngram`, `mtp`: a role-only SafeTensors file, a flat directory
   with one SafeTensors index and its shards, or a native `.hgn` file for that
   role. Exact tensor names and shapes are required; symbolic links, hard
   links, extra tensors and missing tensors fail.
4. Add the IDs after the source in `input_ids`. Set `options.<kind>` to the
   zero-based index of its ID. For a `verify` job, index 1 is the HGN output,
   so components start at 2. The workflow does this for you.
5. An explicit input replaces the whole embedded role. Other roles stay
   embedded. The full snapshot is still required.

`verify` checks payload integrity and decoded values, and detects a change of
the selected bytes. It does not show that a replacement is from the same
checkpoint, and it does not show quality. Report the origin of each
replacement only if the user gave it.

### Custom profiles

A custom profile moves chosen tensors between the two built-in encodings
(`v2`, `ht43`). Use it only when the user asks.

1. Check that the host reports `capabilities.custom_profiles: true`. If not,
   stop and report the `reasons` entry. A custom profile needs an `hf`
   source and `cpu` for every backend, including `decode_backend`.
2. `GET /api/profiles/catalog`. Take `catalog_sha256`. Pick only tensors with
   `editable: true`.
3. `POST /api/profiles` with `name`, `base`, `catalog_sha256` and `overrides`
   (tensor to the encoding that differs from `base`). `201` is new, `200` is
   the same content. Keep the `revision`; it never changes.
4. Submit the conversion with top-level `profile_revision` and no
   `options.profile`. `404` is an unknown revision, `409` an incapable host.

Report three things separately. `size.predicted_bytes` is a layout estimate
from the catalog, not a measured size. A passed `verify` shows that the file
matches the saved recipe, with source provenance, payload and decoded-value
checks. Runtime accuracy and model quality are unproven until `generate` or
`evaluate` jobs run, and even then the result is not a release qualification.
The result has `custom_profile.release_qualified: false`.

### GGUF output

Add `"format": "gguf"` only when the user asks for GGUF. Without `format`,
the HGN workflow runs as before. Do not mix the two: a saved profile must
have the same format as the conversion.

1. Check the host `capabilities`: `gguf_output` for all sources, plus
   `gguf_hf_output` for HF or `gguf_safetensors_output` for standalone
   SafeTensors. If a required capability is false, stop and report the `reasons`
   entry. Do not try to enable it.
2. `GET /api/profiles/gguf-catalog`. Use only a `preset` or tensor type from
   it. There is no `IQ1_XXS` and no F32 preset. Do not invent a name.
3. For a floating-point GGUF or supported HF source, `input_ids` starts
   with the source. For standalone SafeTensors, start with the weights and
   a separate matching `assets` directory (config and tokenizer) on the same
   worker. Append at most one registered `imatrix` as the last input. An `hgn`
   source is refused. If the preset or type has `requires_imatrix: true` and
   the user gave no imatrix, ask; do not guess one.
4. Options are only `preset` and `threads`. To mix types, save a profile
   with `POST /api/profiles` and `"format":"gguf"` (`preset`, up to 256
   exact-tensor `overrides`), then send `profile_revision`, not `preset`.
   Floating-point presets take no overrides.
5. The workbench cannot yet publish Flash-Next as a verified GGUF pair.
   The bounded exporter supports specific dense Qwen2 and GPT-2 layouts;
   check the architecture and tokenizer limits in `OPERATOR.md`. Report a failure; do not
   work around it.

Report what the result shows: the output SHA-256 and size, the quantizer
SHA-256, and the tensor types read from the file (`report.gguf_tensors`).
`artifact_model_id` appears only after the independent `verify` passes.
`verify` is a structural check. It does not decode values, load the file in
a runtime, or measure quality. It is not a release qualification.

For an HGN conversion, `artifact_model_id` can exist before `verify` passes. Trust `state`. A
`succeeded` workflow means that four stages passed. It is not generation
quality, GSM8K quality, or a release qualification.

`brewery-agent forget-model ID` removes the library entry only; it never
touches the model file on the worker.

## Settings and administration

Change settings only when the operator asks.

- **Settings.** `GET /api/settings`, then `PUT /api/settings/{host}` with the
  full object and the `revision` that you read. It saves targets (exactly
  `http://127.0.0.1:PORT/v1` or `http://[::1]:PORT/v1`, on an approved port),
  suites from the approved list, baseline endpoints, and the evaluation
  switch where native evaluation is configured. Other defined hosts can save
  target profiles without acquiring permission to run evaluations. On a
  conflict, read the current revision and reason before retrying. Never put a
  secret in a body. A save is not a start: it tests no endpoint and starts no
  job. `ready` is a check of policy and files, not of the model server.
  Confirm the effect in `state` capabilities.
- **Infrastructure.** `GET /api/admin/state` and `POST /api/admin/actions` use
  normal workbench access. A `503` means that no catalog workers are configured
  or that another action is running. Distinguish policy changes from an
  actual installation. For additional SSH hosts, follow `INSTALL.md`: use
  an authorized credential reference, independently verify the host key,
  review the installation plan, and verify the worker after installation.

The CLI has no verbs for these routes; use `curl` with
`Content-Type: application/json`.

## Report

State what ran, with job IDs, final states, and errors. Quote results; do not
average, rank, or upgrade them. Never describe workbench output as a release
qualification. See skill `brewery-release-qualification`.
