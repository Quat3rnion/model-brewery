---
name: brewery-release-qualification
description: Collect and check Model Brewery release evidence. Use for distributed conversion, local sm_120 stages, the six conversion rows, trace replay, Halogen proof, or a release qualification decision.
---

# Release evidence

**Status rule.** The workbench is not release-qualified. Release needs all six
paths and both profiles (`v2`, `ht43`) to pass. Say "not qualified" unless the
checker and the operator's external review say otherwise. `release_check.py`
exits 1 while release or review is outstanding; that is the normal result.
Do not report a development subset, synthetic fixture, benchmark, calibration
report, preview, or Stage 2 receipt as release evidence. Do not claim Halogen
identity, GPU parity, or quality from any output you did not read.

Full requirements: `OPERATOR.md`, "Release qualification". Row procedure:
`qualification/release_matrix.md`. Queue and follow jobs with skill
`brewery-agent-operator`.

## Hard limits

- **Ask the operator before** any full-model job, GPU load, or memory-heavy
  step. They approve the disk, memory, and GPU admission plan.
- **Admission stays.** Run GPU and build work only under the host's GPU
  admission lock and build gate (the worker's `admission` wrapper, or
  `resource-queue`). Do not set `BREWERY_ALLOW_GPU_LOAD`, edit worker config,
  override `CUDA_VISIBLE_DEVICES`, or stop a service to free memory.
- **Never cancel, delete, or replace** another run's jobs, state, or journal.
  Resume by repeating the same command.
- No passwords or tokens in commands, files, or reports.

## Convert, verify, evidence

1. Preview first: `plan` jobs and `transfer_preview` on the intended workers.
2. One `convert` job per row: targets `gfx1151`, `sm_89`, `sm_120` and
   profiles `v2`, `ht43`; explicit `host`, not `auto`. Then a separate
   `verify` job (source first, output second). A conversion receipt is not a
   verify pass.
3. `baseline`, `generate`, `evaluate` (all 1,319 GSM8K rows; candidate at
   most 2 points below baseline); `transfer` any remote bundle and keep that
   job ID.
4. Copy evidence files into `ARTIFACT_ROOT/<job-id>/` (`OPERATOR.md` and
   `release_matrix.md` list them), then collect live records and check:

   ```sh
   python3 qualification/release_matrix.py --exports "$EXPORT_DIR" --artifacts "$ARTIFACT_ROOT" \
     --baseline-job-id "$BASELINE" --row "sm_120/v2=$JOB_ID" $EVIDENCE_JOB_IDS
   ```

   It exports fresh `GET /api/jobs/{id}` snapshots and runs the checker; it
   never queues or retries jobs. Offline form on saved exports:
   `python3 qualification/release_check.py [--row TARGET/PROFILE=ID]
   [--baseline-job-id ID] [--artifacts ROOT] JOB_EXPORT.json ...`. Read
   `upstream_source.status` per row. Unverified GPU use and BF16 provenance
   need operator review.

## Operator CLIs

Check `--help` before use. Each needs operator approval and admission.

| Task | Command |
| --- | --- |
| Local sm_120 stages | `python3 -B qualification/local_sm120.py {preflight,v2-convert,v2-verify,ht43-convert,ht43-verify,cancel,all}`. Repeat a stage to resume; do not delete its journal. `cancel` only when the driver is not running. Covers the two local conversion rows only. |
| Distributed conversion (Stage 2) | `qualification/distributed_conversion.py run --config stage2.json --run-dir RUN_DIR --timeout 3600 --reserved-workers`. Private operator shell, not the HTTP controller. Two distinct reserved workers, shared GPU lock. Resume with the same command and directory; after a terminal failure use a new run directory. Guide: `qualification/distributed_conversion.md`. Not a release result. |
| Trace replay (pinned dense and MoE GGUF) | `python3 qualification/replay_host.py --dense D.gguf --moe M.gguf --out NEW_RUN_DIR --device DEVICE --lanternfish PATH --comparator PATH`. New run directory, under the GPU admission lock. |
| Trace parity | `trace-parity` with fixed-v1 tolerances (`OPERATOR.md`). `--atol` and `--rtol` are diagnosis only. A CPU fallback does not pass. |
| Bundle on disk | `cargo run -p halogen-proof -- qualify /path/to/bundle.json` |
| Halogen file-map proof | `halogen-proof prove ...` loads a full model on the GPU. Needs explicit operator approval, a quiet host, and the gates. See `crates/halogen-proof/README.md`. |

`qualification/reference.md` describes a separate derived-FP32 experiment.
Its results never count toward the native gate.

## Report

For each row: job IDs, states, checker verdict per item, and what is still
missing. Separate "verified by the checker" from "needs operator review".
End with the status: not qualified, or what remains.
