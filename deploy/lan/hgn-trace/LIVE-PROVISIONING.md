# Frame HGN live provisioning

Operator handoff only. Do not run these steps on the live stack without a
separate cutover approval. This change does not qualify a release.

## Worker and access

1. Keep the Hy4 plan and encode workers unchanged. Add a separate
   `frame-hgn-trace` SSH alias, key, and private worker state directory on
   Frame. Use a dedicated, non-root worker identity with read-only access to
   the approved model files. Pin Frame's host key. Restrict the key to
   `brewery-worker rpc` with the new `BREWERY_WORKER_CONFIG`. Do not allow
   a shell, agent forwarding, or port forwarding.
2. Install `brewery-worker` and `brewery-engine` from the approved release.
   Use a canonical, release-owned path for each binary and worker
   configuration. Set `root` to the new private state directory and
   `source_roots` to the single approved checkpoint directory. Do not use the
   user's entire home or models directory. Do not set `hy4_plan`, `admission`,
   GPU load permission, or any write access to the checkpoint directory.
3. Add a static dispatch entry for `frame-hgn-trace` that uses the new SSH
   alias. Explicitly disable `inspect`, `plan`, `convert`, `verify`,
   `compare`, `transfer`, `transfer_preview`, and `cache_cleanup`: these
   operations otherwise default to on. Enable `trace` only after the checks
   below. Enable `behavior_direction` only after you install the reviewed
   contrast cache on the worker and set `BREWERY_CONTRAST_CACHE` in its
   `engine_env`. The HGN screen needs both capabilities and uses the
   version 2 direction from the same Library checkpoint. Keep general
   weight jobs disabled; `find_circuit` still requires GGUF.
   Set CPU and RAM bounds on both the controller and the worker. Use no
   quantization pool. Start with 16 CPU threads and at least 16 GiB of
   available RAM, subject to the Frame operator's admission limits.

## Enroll the approved model

Follow [HGN CPU setup](../../../OPERATOR.md#hgn-cpu-setup) for the pinned
CPU venv. Put `assets.json` in the release directory on **Frame**, not in
the worker state directory. Set `BREWERY_HGN_TRACE_PYTHON`,
`BREWERY_HGN_TRACE_PYTHON_SHA256`, `BREWERY_HGN_TRACE_ASSETS`, and
`BREWERY_CPU_THREADS` in the new worker's `engine_env`. The worker must not
have write access to the release files or permission to replace them.

Record the embedded CPU runtime SHA-256
`1ac8e30b17ccd7b96ff84f840c3c6d911cb2cbded57d6a5cfec86217c417aef7`
from `tools/hgn_trace/runtime.sha256` in the release manifest. Use matching
controller, dispatcher, worker, and engine builds. The runtime digest does
not replace the interpreter or asset digests. Published version 2 direction
receipts remain readable, but new fits must use the release runtime.

The registry key is the verified checkpoint SHA-256. Each member has an
absolute canonical file path and a lowercase SHA-256. For ThinkingCap, the
reviewed source directory on Frame is
`/home/quaternion/models/thinkingcap-hgn-dbe4b52`.

| Registry field | File | SHA-256 |
|---|---|---|
| Key | `model.hgn` | `5635d055a9e3b4ecefed585bfdcd0b62a20b9b388ab268eed35e768d6b924835` |
| `sidecar` | `qwen38-flash-next-ngram.hgn` | `339f5c508ca592ee2613ac5c3082587a88991d5592c4160d49a32a6dbdb3f30b` |
| `config` | `config.json` | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| `tokenizer` | `tokenizer/tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

Check all four hashes under the Frame build reservation before you enable
trace. Compare the large file hashes with the source manifest. Record the
registry, executable and config digests in the release manifest. Do not copy
the model or the sidecar to the controller. Do not write to the model
directory. If a pin changes, review and install a new registry; do not
update a deployed registry to match unexplained bytes.

Register the `.hgn` checkpoint as `hgn-v2` on `frame-hgn-trace`. Use that
Library ID in `start_trace`. Keep the existing record on `frame-hy4-plan`.
Sending its ID with a different `host` does not move the source.

## Optional private edits and checks

Keep `scratch_hgn_ablation` disabled until the operator approves private edit
storage, a signing key, and build admission. Follow [Native HGN v2 scratch
edits](../../../OPERATOR.md#native-hgn-v2-scratch-edits). Use the same worker
for the stock checkpoint and its private children. Set `hgn_edit.host` to
the dispatch ID, `hgn_edit.grant_key_file` to the worker's private key file,
and `hgn_edit.build_admission` to the approved exclusive CPU build wrapper.
Set `BREWERY_HGN_EDIT_KEY_FILE` on the controller to its private copy of the
same key. Do not put a key value in the release manifest. Keep `trace` enabled
for `run_checks`: it uses the pinned CPU runtime for the paired forward.
Do not register or publish native scratch files.

## Quiet window, verification and rollback

- Coordinate with Frame operators. Keep this worker paused outside the
  approved window. For trace, direction fitting, and screening, hold
  Frame's exclusive `build` reservation from before enqueue until the job
  is terminal and the worker is idle. The reservation must cover source
  hashing and inference. For private edits and stored checks, let the
  `build_admission` wrapper acquire the reservation. Do not hold an outer
  reservation for those jobs. A reservation timeout alone does not stop the
  detached Brewery supervisor; request cancellation and confirm that it
  has stopped. Do not start a Frame GPU job for these checks.
- Before cutover, run `qualification/hgn_trace_job.py` on a new isolated
  stack on Frame with the release's binaries, reviewed registry, and pinned
  interpreter. Check the model and asset digests, `hgn_cpu` receipt, four
  residual streams, final normalized row, and identical final lens and
  native top logits. Run `qualification/hgn_trace_parity.py` with the
  matching saved Halogen rows; do not request new rows from the production
  service.
- For cutover, use the controller release's normal idle fence and backup
  procedure. Verify the alias, effective capability, and Library path before
  use. Do not alter the production Halogen service or another Frame worker.
- To roll back, pause the new worker, wait for it to become idle, and remove
  its HGN capabilities through the normal release configuration procedure.
  Preserve both Library records and the worker artifacts for audit. Only
  retire the new SSH key and worker after no process or job needs it.
