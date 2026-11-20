---
name: brewery-calibration-benchmarks
description: Run and read Model Brewery endpoint benchmarks, scorecard export, offline calibration reports (expert usage, profile preview, Halogen speculation), and build or paired-quality comparison. Use for benchmark setup, tools/export-scorecard, or the Calibration page. Not for release claims.
---

# Benchmarks, scorecards, calibration

All output here is diagnostic. None of it qualifies a release, and none of it
is a calibration or quality result unless a script says so.
Queue and follow jobs with skill `brewery-agent-operator`. Operator setup
and field details: `OPERATOR.md`, "Managed benchmarks" and "Offline usage and
profile preview".

## Endpoint benchmarks

1. **Check the host.** In `state`, the host needs `capabilities.benchmark`
   `true` and non-empty `benchmark_targets` and `benchmarks` (each `{id,
   label?}`). If not, report the `reason`. The operator enables it with
   `engine_env.BREWERY_BENCHMARK_CONFIG` in the worker file and trusted
   absolute adapter executables. Do not write or edit that config, install
   adapters, or point a target at a non-loopback endpoint.
2. **Submit** a `benchmark` job: no inputs, `options` with only
   `benchmark_target` and `benchmarks` (1 to 8 distinct IDs):

   ```sh
   tools/brewery-agent create jobs --key bench-1.v1 --file bench.json
   tools/brewery-agent wait "$JOB_ID" --timeout 1800
   ```

   Suites run in order, 300 s each. Managed protocol IDs: `tool-eval-safe-v1`
   (`tool-eval-bench` 3.0.0, TC-01/02/04, mock tools only) and
   `betterbench-decode-chat-v1` (`betterbench` 0.6.1, 1 warmup and 4 chat
   requests). Another ID is an operator adapter; read its `protocol` and
   `version` in the report, do not assume.
3. **Read the report** (`result`, schema `brewery-benchmark-report-v1`):
   `status` `complete` | `partial` | `failed`; each run has `status`,
   `metrics` (`name`, `value`, `unit`, `direction`), `samples`, `error`,
   `raw_artifact`. One failed suite does not stop the next. `release_qualified`
   is always `false`.

## Scorecard

Save each job detail unchanged, then export offline (needs Bun on PATH; no
network):

```sh
tools/brewery-agent job "$JOB_ID" > detail-$JOB_ID.json
tools/export-scorecard detail-*.json > scorecard.json
```

Only `benchmark` and `evaluate` details; 1 to 256 files, 16 MiB each, no
duplicate IDs. An invalid input exits nonzero and writes nothing. Display
host names are `null` (not in job details). The scorecard lists
`failed_benchmarks` and `missing_benchmarks`: read both for coverage. Do not
add, average, or rank metrics across suites, jobs, or models. Compare runs
only with the same target, model, version, and protocol.

## Calibration scripts (offline, local, read-only)

They load no model and call no controller. Files stay local; do not upload
production traces or plans. `--help` on each is the source of truth.

| Question | Command | Output |
| --- | --- | --- |
| Which experts did saved routing select? | `python3 qualification/expert_usage.py TRACE [TRACE ...] > usage.json` | `brewery-expert-usage-v1` |
| Size of a whole-tensor profile swap? | `python3 qualification/profile_preview.py --recipe recipe.json --v2-plan v2.json --ht43-plan ht43.json > preview.json` | `brewery.profile_preview.v1` |
| Draft acceptance counts from Halogen timings? | `python3 qualification/halogen_speculation.py --producer-version 0.18.22 [--wrapped] capture > speculation.json` | `brewery-halogen-speculation-v1` |

- **Traces** are Lanternfish replay JSON v1. Limits: 64 files, 64 MiB and
  512 tokens per trace; limit violations fail, never truncate. Traces are
  sparse: `missing_positions` marks absent routing, a layer with none has
  empty `experts`, zero weight is a value, not missing. Counts describe one
  workload. They show neither importance nor accuracy and do not justify
  removing or quantizing an expert.
- **Plans** come from the operator's converter: `bf16-to-hgn plan --source
  "$SOURCE" --profile v2 > v2.json` and `--profile ht43 > ht43.json`, same
  source and options. Recipe: `{"schema_version":1,"base_profile":"v2",
  "overrides":[{"select":{"layers":[0],"role":"hyper_connection"}}]}`.
  `select` takes `layers` and/or `role` (`hyper_connection`, `routed_expert`)
  or exact `tensors`. There is no codec field. Swaps are whole tensors and
  carry labeled companions. The result is an estimate with
  `executable:false`, `release_eligible:false`, and unverified weight
  identity. It is not a runnable profile or a quality claim.
- **Viewer.** The web page's **Calibration** tab (`#/calibration`) opens
  these three schemas from local files (16 MiB each). Nothing is uploaded.
- Held-out sensitivity and per-profile quality are future work. Do not
  report a preview or usage count as one.

## Comparison CLIs

Check `--help` first; if a script is missing in this checkout, report that
and stop.

```sh
python3 qualification/build_diff.py LEFT_CONVERT_JOB.json RIGHT_CONVERT_JOB.json
python3 qualification/paired_quality.py LEFT_JOB RIGHT_JOB \
  --left-predictions FILE --right-predictions FILE
```

Inputs are unchanged `GET /api/jobs/{id}` records. Use them only for matched
workloads: paired questions on the same protocol. A difference is a
comparison, not causal attribution. Report cross-model or cross-version
differences as such, apart from any quantization difference.
