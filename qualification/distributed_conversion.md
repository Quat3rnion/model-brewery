# Stage 2: Distributed conversion

Run this tool on the controller from a private operator shell. It does not
submit to the Stage 1 HTTP controller. It uses the same `brewery-dispatch`
and `brewery-worker` programs and their allowlisted local or SSH transport.
Each worker must use a dedicated Stage 2 configuration.

## Before you run

- Reserve the listed workers for this run. Two distinct worker IDs are
  required. Do not give two IDs to the same worker or GPU. The source may
  be on a different allowlisted host; for large sources, keep it on the
  coordinator to avoid an extra transfer.
- On each participating worker, set `engine` to the absolute path of the
  executable `qualification/distributed_conversion.py`. Set
  `engine_env.BREWERY_STAGE2_NATIVE` to the absolute path of the installed
  shard-capable `bf16-to-hgn` binary. The tool requires Python 3.11 or newer.
- For GPU work, configure `admission` on the worker to use the **same
  exclusive lock** as Stage 1 and other jobs on that GPU. Use the installed
  admission gate or a site wrapper compatible with the worker's
  `--wait -- ENGINE` interface. Pin the wrapper file's SHA-256 below.
  Preflight checks the configured file's digest and runs through that
  wrapper. It cannot tell if two different wrappers share a physical lock;
  the operator must verify this. A separate worker state root is not a lock.
- Make the original HF snapshot available under a configured worker source
  root. Keep the snapshot and work outputs immutable. Allow enough space
  for the source, native packages, worker caches, unit outputs, and merged
  model. The workflow retains all work directories. The worker cache and
  native stages enforce their own free-space checks.

## Configuration

Use an operator-owned JSON file. This example assumes the installed
`dispatch.json` already lists the three named hosts. Replace the placeholder
digests with lowercase SHA-256. `source.sha256`, if given, is the worker's
bundle digest, not the digest of an individual safetensors file.

```json
{
  "dispatcher": "/opt/brewery/bin/brewery-dispatch",
  "dispatch_config": "/etc/brewery/stage2-dispatch.json",
  "source": {"host": "coordinator", "path": "/models/hf/source"},
  "coordinator": "coordinator",
  "workers": {
    "coordinator": {
      "native_sha256": "<coordinator binary SHA-256>",
      "backend": ["cpu", "cpu", "cpu"]
    },
    "encode-a": {
      "native_sha256": "<worker A binary SHA-256>",
      "backend": ["cpu", "cpu", "cpu"]
    },
    "encode-b": {
      "native_sha256": "<worker B binary SHA-256>",
      "backend": ["cpu", "cpu", "cpu"]
    }
  },
  "encoding": {
    "profile": "ht43",
    "seed": 0,
    "threads": 1,
    "ht-beam": 128,
    "ht-backend": "cpu",
    "i4r-backend": "cpu",
    "q4cp-backend": "cpu"
  },
  "decode_backend": "cpu"
}
```

An encoding worker must match the exact `backend` tuple, in HT, I4R, Q4C-P
order. A coordinator with a different tuple can partition, pack, gather,
and merge without encoding. Set each GPU worker's `admission_sha256` to
the installed shared gate's digest. The accepted Q4C-P backends are
`cpu` and `cuda` in this interface.

## Run or resume

```sh
qualification/distributed_conversion.py run \
  --config /path/to/stage2.json \
  --run-dir /arboretum/stage2-run \
  --timeout 3600 --reserved-workers
```

Use an absent or private (`0700`) run directory. Run the same command with
the same configuration and run directory to resume after a timeout or
transport failure. The tool locks the journal, keeps stable job IDs,
and checks completed receipts before it continues. It does not cancel
work when the client exits, reassign a running unit, or retry a failed
terminal job. Inspect the worker's `engine-stderr.log` and use a new run
directory after you correct a terminal failure. Keep the old run directory
and worker outputs as evidence.

The tool starts at most one encode job per eligible worker in each batch.
It sends only the partition and each unit's own package to its encoding
worker. Collection jobs gather larger sets without exceeding the worker's
128-input limit. A result must match the recorded partition, unit ID and
artifact digest. The native merge must pass its own provenance check
before the worker publishes the artifact. On success, `result.json` gives
the coordinator hostname, artifact path and digest, and native merge receipt.

This result is **not a release qualification**. In particular, the workflow
does not produce a Halogen runtime proof, GSM8K result, GPU decode parity
result, Stage 1 runtime wrapper, or release admission. Do not use a tiny
fixture's receipt as full-model evidence.

## Focused tests

```sh
TMPDIR=/arboretum CARGO_TARGET_DIR=/arboretum/stage2-integration-target \
  resource-queue run --resource build --holder brewery-stage2 \
  --max-hold 90 --wait 90 --flock build -- cargo build --offline -p brewery-worker
BREWERY_TEST_BIN=/arboretum/stage2-integration-target/debug \
  python3 qualification/test_distributed_conversion.py -v
```

These tests run two independent local Rust worker configurations through
the real dispatcher. Their native CLI test double uses clearly synthetic
data. They cover interrupted work, resume without duplicate work, and
rejection of a changed unit before publication.

For the optional native fixture test, set `BREWERY_TEST_NATIVE_FIXTURE`
to the absolute path to the native development harness. The harness must
support `fixture --root NEW_DIR --profile v2|ht43` and the same shard CLI
as the production converter. The test uses real shard encoding, merge,
and verification, and checks both profiles against the serial reference.
It uses fixture plans, not the production architecture planner.
