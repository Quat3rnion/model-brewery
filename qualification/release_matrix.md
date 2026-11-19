# Collect evidence for the six conversion paths

## Local sm_120

On the approved local workstation, use the prepared worker under
`/mnt/games/model-brewery/local-sm120`. The worker references the published
snapshot in place. Do not register or use the partial download.

```sh
python3 -B qualification/local_sm120.py preflight --wait-publication 21600
python3 -B qualification/local_sm120.py v2-convert
python3 -B qualification/local_sm120.py v2-verify
python3 -B qualification/local_sm120.py ht43-convert
python3 -B qualification/local_sm120.py ht43-verify
```

Alternatively, `python3 -B qualification/local_sm120.py all --wait-publication 21600`
runs these stages in order and stops at the first failure. Conversion
requires both plans, space for the remaining outputs, at least 16 GiB
available host RAM, at least 12 GiB free VRAM on the pinned GPU, and a
shared GPU lease. These memory floors are provisional guards, not a
capacity guarantee. Source and output hashing do not hold the GPU lease.

To resume after losing contact, repeat the same command with the same
worker state. Do not delete the journal or submit replacement jobs. If the
driver cannot establish the outcome of a POST, it stops. To stop pending
work, interrupt the foreground driver or run `local_sm120.py cancel` once
the driver is no longer running. The driver shuts down its controller at
the end of each stage.

Use the raw controller responses in `local-sm120/exports` and the
corresponding files in `local-sm120/artifacts` with `release_check.py`.
This procedure covers only the two local conversion rows and their file
checks, not inference, the BF16 baseline, or strict trace parity.

## All targets

Do not start full-model jobs until the operator approves the disk, memory,
and GPU admission plan. Run transfer previews and converter plans on the
intended workers first. Small checkpoints do not fill release rows.

1. Register the complete pinned BF16 checkpoint as `hf`. Use the same
   source tree for all six jobs. The checker compares the recorded source
   shards with `qualification/release_check_shards.sha256`. It also checks
   the input index and nine converter assets against the sizes and SHA-256
   values for the pinned revision in `qualification/release_source.py`.
   Look for `upstream_source.status` in each row. A valid conversion record
   alone does not prove that its source matches upstream.

   For the pinned layout with exactly 131 shards and the ten metadata files
   (no cache, README, or receipt files), the expected size is
   360,023,281,554 bytes. The expected `bundle_sha256` is
   `0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1`;
   the role-qualified `source_sha256` for BF16 baseline matching is
   `8858fcd3bad25db0f243a0f72d30ad0674f4a99bf04d102ab61f01897df72308`.
   These are predictions from the immutable upstream file records. Confirm
   both against the worker's actual input record. Do not treat them as a
   receipt that all files are present or that an endpoint served the weights.
2. Create one `convert` job per pair: `gfx1151`, `sm_89`, and `sm_120`, each
   with `options.profile` set to `v2` and to `ht43`. Set the controller
   `host` to the worker for that target, not `auto`. Set `ht_backend` to
   `hip` on gfx1151 or `cuda` on NVIDIA. Set other codec backends as
   applicable to the pinned converter. Keep all six job IDs. For each job,
   capture independent GPU and encoder kernel evidence tied to its worker,
   executable, process, run interval, and output hashes.
3. For each succeeded conversion, register `result.artifact.path` on the
   producing host as `hgn`. Create a separate `verify` job with the source
   first and that output second. Check for an explicit pass. A conversion
   receipt does not replace `verify`.
4. Create a `baseline` job with `options.baseline_source` set to the
   configured BF16 endpoint for this checkpoint. Confirm independently that
   it served the exact BF16 weights. Register its artifact as
   `gsm8k-baseline`. On the qualified Halogen host, run `generate` with a
   saved prompt and `evaluate` with the HGN bundle first and the baseline
   file second. For remote HGN bundles, run `transfer` to the inference
   host, register the transferred copy, and retain the successful transfer
   job ID. A preview is not transfer evidence. Reuse inference evidence
   only for matching file hashes, runtime, and protocol. Use all 1,319
   GSM8K test rows. The candidate may score at most two percentage points
   below baseline.
5. Copy these files to `ARTIFACT_ROOT/<job-id>/` using their original
   basenames. Use the job ID that produced or references each file:

   - Convert: `native/conversion-manifest.json`, `manifest.json`, and the
     input `model.safetensors.index.json`. For an observed CUPTI capture,
     also copy `encoder-activity.json`, the native converter executable,
     and the collector library. The checker hashes the binaries; it does
     not run them. Confirm the worker inventory independently. A trace
     of one fixture does not prove that a full-model job used the GPU.
   - Verify: the referenced `verify.json`.
   - Baseline and evaluate: the referenced prediction JSON artifacts.

   Collect the live controller records using the job UUIDs:

   ```sh
   python3 qualification/release_matrix.py --exports "$EXPORT_DIR" --artifacts "$ARTIFACT_ROOT" \
     --baseline-job-id "$BASELINE" \
     --row "gfx1151/v2=$AMD_V2" --row "gfx1151/ht43=$AMD_HT43" \
     --row "sm_89/v2=$SM89_V2" --row "sm_89/ht43=$SM89_HT43" \
     --row "sm_120/v2=$SM120_V2" --row "sm_120/ht43=$SM120_HT43" \
     $EVIDENCE_JOB_IDS
   ```

   `EVIDENCE_JOB_IDS` is a space-separated list of verify, transfer,
   generate, and evaluate job UUIDs. Use a private `EXPORT_DIR`. Set
   `--controller` if it does not listen at `http://127.0.0.1:4141`.

Run the same command again to refresh the snapshots. Omit unknown job IDs
and their `--row` or `--baseline-job-id` options. The command does not queue
or retry jobs or substitute old snapshots for missing jobs. It writes the
checker report to stdout and collection status to stderr. Unverified GPU
use and BF16 provenance require operator review. This check does not
replace strict trace parity or qualify a release by itself.
