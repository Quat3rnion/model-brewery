# Model Brewery

Model Brewery is a workbench for one developer. A Rust service and a Svelte
web page run on one Linux workstation (the controller). The controller sends
jobs to workers over SSH. It keeps one model library and one job queue for all
hosts.

The workbench does these tasks:

- Trace and compare two GGUF models with Lanternfish for one saved prompt.
- Convert a supported Qwen3.8 Flash-Next checkpoint to HGN v2 or HT43 with
  the bf16-to-hgn converter.
- Verify the output, generate text, and score the pinned GSM8K protocol.
- Move verified model bundles between hosts.

HGN tracing, dataset preparation, and fine-tuning are not in v1.

> **Release status: not qualified.** The workbench runs, but native trace parity,
> full-size conversion, and model quality have not passed the release gates.
> See [Release qualification](#release-qualification).

## Contents

- [Crates](#crates)
- [Requirements](#requirements)
- [Build](#build)
- [Run the controller](#run-the-controller)
- [Set up workers](#set-up-workers)
- [Operator paths and environment](#operator-paths-and-environment)
- [Job flow](#job-flow)
- [Sample JSON](#sample-json)
- [Agent interface](#agent-interface)
- [Limits and behavior on failure](#limits-and-behavior-on-failure)
- [Release qualification](#release-qualification)
- [Smoke test](#smoke-test)
- [Managed benchmarks](#managed-benchmarks)
- [Web settings and administration](#web-settings-and-administration)
- [Offline usage and profile preview](#offline-usage-and-profile-preview)
- [Trellis weight formats](#trellis-weight-formats)
- [Weight edit tools](#weight-edit-tools-fixtures-only)

## Trellis weight formats

A bit width does not identify a codec. Match the stored tensor type, its
companion data, and the reader. Use the table to identify a format, not to
infer that a worker can write it.

| Family | What identifies the weights |
| --- | --- |
| Native HGN | `HT3` and `HT4` use dtype 16 with variants `0x1206` and `0x1208`, respectively. Both use `Su` and `Sv` companions and require matrix dimensions divisible by 128. `HT43` is a model profile, not a third tensor type. `ht_beam` sets the encoder search width, not the weight precision. |
| [ik_llama.cpp KT](https://github.com/ikawrakow/ik_llama.cpp/blob/7c00f8788e303bfba2886138e288effb0fd5fbbf/ggml/include/ggml.h) | Fork tensor IDs are `IQ1_KT` = 158, `IQ2_KT` = 153, `IQ3_KT` = 154, and `IQ4_KT` = 155. Do not use model preset numbers as tensor IDs. These [sequence-generator formats](https://github.com/ikawrakow/ik_llama.cpp/pull/113) are not QTIP checkpoints. |
| [Gyro Trellis GGUF](https://gitlab.com/agentionai-group/agention-gyro/-/blob/587a5269848eb07996c9c1433f9ad638604eb4d8/SPEC.md) | Agention specifies eight layouts: `TQK4` through `TQK7`, `TQ2_T`, and `TQP5` through `TQP7`. Its public reference encoder is an uncalibrated baseline, not the encoder used for its published models. The [ik reader integration](https://github.com/ikawrakow/ik_llama.cpp/pull/2812) supports six of the eight layouts, not `TQK4` or `TQK5`, and remaps their disk IDs on load. Keep the declared Hadamard rotation with the weights. Stock GGUF `TQ1_0` and `TQ2_0` are **ternary**, not trellis formats. |
| [QTIP](https://github.com/Cornell-RelaxML/qtip), [EXL3](https://github.com/turboderp-org/exllamav3), [GLQ](https://github.com/cnygaard/glq) | These projects have distinct checkpoint layouts and readers. Keep codebooks, transforms, and scales with their packed weights. A target average bitrate is not the stored type of every matrix. GLQ's trellis path is distinct from its E8 lattice path. |

For a layer override, check the actual stored type after export and decode it
with a matching reader. Header inspection alone does not establish that a
runtime can load the model. A successful decode does not establish model
quality. Do not substitute another quantization if the requested writer,
matrix geometry, companion data, or runtime is unavailable.

For the native HGN layer recipe tool, build the CPU adapter from its locked
manifest. The build needs a C++17 compiler and OpenMP (`libgomp` on Linux):

```sh
cargo build --locked --manifest-path tools/cross_container/hgn_in_gguf/Cargo.toml \
  --bin brewery-layer-quant-hgn
```

Pass the resulting executable to `tools/layer_recipe_hgn.py --helper`.
The inputs are a dense Llama or Qwen2 SafeTensors checkpoint, its
source-bound recipe, and a new output directory. For a standalone file,
also pass `--assets` with its matching configuration and tokenizer.
Do not use this path for a fused rank-3 Flash-Next expert matrix.
The receipt's `runtime_load_verified` field distinguishes a verified
archive from a model that has run in Halogen.

To enable managed native HT3/HT4 recipes in Convert, install the same locked
helper on a local source worker. Set `BREWERY_LAYER_HGN_HELPER` to its absolute,
symlink-free executable path and `BREWERY_LAYER_HGN_HELPER_SHA256` to its
lowercase SHA-256 in the worker's `engine_env`. The engine uses the system
`/usr/bin/python3` and its bundled exporter. The helper must support `identity`,
`plan`, `write`, and `inspect`; rebuild older helpers. Set `capabilities.hgn_layer_recipe`
to `true` for this local worker in the controller dispatch configuration.
Do not enable this capability on a worker that does not have the pinned helper
or that uses a pinned publication or `BREWERY_EXPECTED_SOURCE_SHA256`.

The Convert screen also needs read-only SafeTensors inventory on the **same
worker**. [Install the local CPU weight runtime](#package-the-local-cpu-weight-runtime).
In `engine_env`, set `BREWERY_WEIGHT_EDIT` to the absolute, symlink-free path to
the packaged `brewery-weight-edit`, set `BREWERY_WEIGHT_VISUAL` and
`BREWERY_WEIGHT_SAFETENSORS` to `"1"`, and set
`BREWERY_WEIGHT_SAFETENSORS_VERIFY_BYTES` to `"67108864"` for bounded sources
without a publication proof. Set `capabilities.weight_inspect`,
`capabilities.weight_visual`, and `capabilities.weight_safetensors` to `true`.
You do not need to enable `weight_propose`, `weight_apply`, or `weight_tune`.
The conversion API can run with a known digest without the inventory helper.
Register a changed checkpoint as a new Library entry so Convert reads a new
digest and layer list instead of reusing the previous entry's inventory.

Select Layer recipe and HGN archive in Convert. Use an indexed dense Llama or
Qwen2 HF source, or register standalone SafeTensors and matching config and
tokenizer assets on the same worker. Each source input is limited to 64 MiB.
Choose HT3 or HT4 for at least one layer. The worker checks the source digest,
asset digests, helper identity, and geometry before it encodes. No calibration
input is used. A changed recipe produces a new archive and does not modify the
source or previous output.

The controller registers an `hgn-archive` only after a separate verify stage
reopens the archive and checks its files and tensor layout. It is not a runnable
Halogen model, an EXL3 checkpoint, or a reference QTIP checkpoint. Do not use
verification as a quality or release claim. Fork GGUF KT recipes are not enabled
by this capability. Preflight includes the archive that stays on disk and the
temporary verification copy. Verify checks the space for its copy, not for a
second conversion. The check does not reserve space.

## Crates

| Crate | Binary | Purpose |
| --- | --- | --- |
| `crates/controller` | `brewery` | HTTP API, job queue, SQLite state, and web page host. |
| `crates/worker` | `brewery-worker`, `brewery-dispatch` | Worker RPC, file transfer, job lifecycle, and cancellation. |
| `crates/engine` | `brewery-engine` | Trace comparison and converter inspect, plan, convert, and verify. |
| `crates/eval` | `brewery-eval` | Halogen generation and the pinned GSM8K run, scoring, and baseline. |
| `crates/halogen-proof` | `halogen-proof` | Bundle qualification and an isolated check of which files the Halogen engine maps. |
| `crates/trace-parity` | `trace-parity` | Fixed-token CPU/GPU trace comparison. |
| `web/` | none | Svelte 5 and Vite web page. Output goes to `web/dist`. |
| `qualification/` | none | Replay inputs, replay scripts, and the workbench smoke test. |

## Requirements

On the controller:

- Linux.
- A Rust toolchain that supports edition 2024 (Rust 1.89 or later).
- Node.js 22.12 or later with npm, for `web/`.
- Python 3, for the scripts in `qualification/`.
- An SSH client and an SSH config alias for each remote worker.

On each worker:

- The `brewery-worker` and `brewery-engine` binaries.
- A private state directory.
- The pinned Lanternfish (llama.cpp) build, for GGUF tracing.
- The pinned bf16-to-hgn converter build, for conversion. The binary must
  match the worker GPU (for example, sm_89 or sm_120).
- For `generate` and `evaluate`: Docker and a trusted Halogen image that is
  already installed on that host. The worker never pulls an image.

The workbench does not infer model identity from file names. Pin each source
model, weight file, tokenizer or asset file, Lanternfish build, and converter
build by hash.

## Build

Run these commands in the project root.

```sh
cargo build --workspace --release --locked
npm --prefix web ci
npm --prefix web run build
```

The Rust binaries go to `target/release/`. The web page goes to `web/dist/`.

Run the tests:

```sh
cargo test --workspace --locked --offline
npm --prefix web run check
npm --prefix web test
```

## Run the controller

Create a private state directory and a dispatch file (see
[Sample JSON](#sample-json)). Then start the controller:

```sh
export BREWERY_DATA=/abs/path/brewery-state
export BREWERY_DISPATCH_CONFIG=/abs/path/dispatch.json
export BREWERY_DISPATCH=/abs/path/to/target/release/brewery-dispatch
export BREWERY_WEB_DIST=/abs/path/to/web/dist
target/release/brewery
```

The controller listens on `127.0.0.1:4141` by default. Open
`http://127.0.0.1:4141/` in a browser on the same machine. Set
`BREWERY_LISTEN` to use a different loopback address and port. Do not expose
the controller on a public interface: it has no login. See
[Web settings](#web-settings) for infrastructure controls.

The optional HTTPS gateway checks the client's source address, the exact
Host, and browser request headers before it forwards requests to the
loopback controller. The gateway does not ask for a login. Allow only
trusted client networks: clients on these networks have controller
operator permissions. The gateway does not bypass worker permissions or
GPU admission. Without endpoint overrides, the gateway binds to
`127.0.0.1` and accepts only `127.0.0.1` clients and `Host: 127.0.0.1:8443`.

For LAN access, add a user systemd drop-in for `model-brewery-lan.service`:

```ini
[Service]
EnvironmentFile=%h/.config/model-brewery/lan.env
```

Create the environment file with your own interface address, allowed client
CIDRs, and client-facing hosts. These values are examples only:

```ini
BREWERY_BIND=192.0.2.10
BREWERY_CLIENT_NETS=192.0.2.0/24
BREWERY_HOST=brewery.example.invalid
BREWERY_ALT_HOST=192.0.2.10
```

Separate multiple bind addresses or client CIDRs with spaces. Set the two
host values without `:8443`; use the same value for both if you have no
alternate. Use brackets around an IPv6 literal in a host value, but not in
`BREWERY_BIND`. The HTTPS certificate must cover the names or addresses
that clients use.

Transfer the public `~/.local/state/model-brewery-lan/tls/lan-ca.crt` to
your client over a trusted channel. Trust that certificate on the client.
Then open `https://<configured-LAN-host>:8443/`. Do not copy
`tls/server.key`.

For Tailscale, `model-brewery-tailnet.service` requires
`~/.config/model-brewery/tailnet.env`. Set the same four variables for your
tailnet: use this machine's Tailscale addresses for `BREWERY_BIND`, the
allowed tailnet client CIDRs for `BREWERY_CLIENT_NETS`, and the Tailscale IP
or DNS name that clients use for each host value. Copy the public certificate
`~/.local/state/model-brewery-lan/tls/tailnet-ca.crt` from the host over
a trusted channel. Trust it on the client. Then open the configured tailnet
host on HTTPS port `8443`. Do not copy any `.key` file. The tailnet gateway
does not use public Funnel access. To remove only tailnet access, run
`systemctl --user disable --now model-brewery-tailnet.service`.

Check the services with `systemctl --user status model-brewery-lan.service
model-brewery-lan-bridge.service model-brewery-lan-standby.service`. Stop
the gateway to remove LAN access without stopping the active qualification
job. The standby waits for the workflow lock; after qualification it serves
the same library and jobs. Stop standby before starting a new
`local_sm120.py` workflow; start standby again after the workflow has
acquired its lock. During controller handoffs the gateway can return `503`.

For a gateway `502`, compare its timestamp with the bridge journal:

```sh
journalctl --user -u model-brewery-lan-bridge.service --utc \
  --since '15 minutes ago' -o short-iso-precise --grep=brewery_bridge_failure
```

`tcp_connect` means the bridge was connecting to the controller. `verify_peer`
means TCP connected but the bridge was checking which process accepted it.
For that phase, `peer_check` is the last check result, and `connect_ms` shows
how much of the shared one-second deadline the TCP connection used. Relay
failures include byte counts for each direction. Records describe connections,
not individual HTTP requests. Use the gateway journal for the route. The
bridge does not log URLs, headers, bodies or credentials.

The workbench follows the Halogen Dashboard visual style. Use the shared
colors, typography, and control styles in `web/src/app.css` for new views.

For web page development, run `npm --prefix web run dev`. Vite serves the page
on port 5174 and proxies `/api` to the controller. Set `BREWERY_CONTROLLER`
if the controller uses another address.

## Set up workers

A worker is a host alias in the dispatch file. There are two kinds.

### Local worker

A local worker runs on the controller host. Give the absolute path of the
worker binary and the worker configuration file:

```json
{ "local": true, "worker": "/abs/brewery-worker", "config": "/abs/worker.json" }
```

### SSH worker

An SSH worker runs on another host. Give the SSH config alias:

```json
{ "ssh": "frame" }
```

1. Install `brewery-worker` and `brewery-engine` on the worker. Make sure
   the SSH login can find `brewery-worker`.
2. Create `~/.config/model-brewery/worker.json` on the worker, or set
   `BREWERY_WORKER_CONFIG` to another path.
3. On the controller, add the host to `~/.ssh/config`. Connect once by hand
   and check the host key fingerprint yourself.

The controller always runs the fixed command `brewery-worker rpc` and sends
JSON on stdin. It uses strict host key checking. A host with an unknown or
changed key fails; do not turn checking off.

### Worker configuration

| Field | Meaning |
| --- | --- |
| `root` | Absolute path of the worker private state directory. Cached copies, job directories, and outputs are here. |
| `source_roots` | Absolute directories that model paths may use. A path outside these directories is refused. |
| `engine` | Absolute path of `brewery-engine`. |
| `admission` | Optional. Absolute path of a GPU admission wrapper (for example `gpu-thermal-run`). |
| `exl3_gpu_queue` | Optional. Absolute path of the host's `resource-queue` executable. Required for managed CUDA and ROCm EXL3 layer jobs. |

For EXL3, the worker runs CUDA and ROCm jobs under an exclusive `gpu`
reservation. The queue can wait up to 30 seconds for a reservation and lets
the engine run for up to 120 seconds after admission. Use the same queue
store and GPU lock configuration as other GPU jobs on that host. Do not
reserve the GPU outside the worker for these jobs. EXL3 does not use the
generic `admission` wrapper; CPU EXL3 does not use either GPU gate.

The worker owns its cache. The workbench never changes or deletes an original
model file. Unregistering a library entry removes only the entry.

## CPU trace jobs

Use `trace` to record selected activations and top logits for a single-file
Library GGUF or HGN, or a scratch revision with private GGUF.
For GGUF, the worker runs the Lanternfish CLI on the CPU. It reads files from the CLI.
It does not start a Lanternfish service or use a Lanternfish HTTP API.
It does not change the model or qualify a release.

1. Install the headless binary from Lanternfish commit
   `0ea30b37729a136a76b8412e7ad1ec91bd7de9b5`. Check its SHA-256:
   `0e4ca38c55a9afa0436eeadd06e89bf295eb201b986dd2f40305f95d3a61e46f`.
   The engine rejects other binaries.
2. Install the matching CPU libraries from llama.cpp `184f653`. Apply the
   patches with `patches/apply-llama.sh` from the pinned Lanternfish commit.
   Put the four files below in an
   operator-controlled release directory. The engine copies and checks each
   file in the private job directory. It sets `LD_LIBRARY_PATH` to those
   copies. It does not need the build directory in the binary's RUNPATH.
3. Add `BREWERY_LANTERNFISH_TRACE` and `BREWERY_LANTERNFISH_TRACE_LIB_DIR` to
   the worker JSON `engine_env`. Set them to the absolute paths of the
   binary and library directory. Add `"trace": true` to the host `capabilities`
   in dispatch JSON. The capability is off by default. Use a regular worker
   with access to the registered GGUF. For scratch, the controller uses its
   local CPU worker. Dedicated download and remote weight editor workers do
   not run traces. The existing GPU `compare` job still uses
   `BREWERY_LANTERNFISH` and its `run` command.

Save this list as `trace-libraries.sha256` in the CPU trace release library
directory. In that directory, run `sha256sum -c trace-libraries.sha256`.
The scratch-check runner uses these same pinned CPU libraries.

```text
bb94467edcc18c8fb875f86628d89c112adedcfea1e8f6c5549c706adf5f25cc  libllama.so.0
04a3a199429b02e6e4bce9a8cbc465530bcffe4b0539be561dedfcfd476957be  libggml.so.0
1fef6e5b16b4b03e8255cf5f3c446cc76005515e0bb261b4d0ee333ea399fbad  libggml-base.so.0
29390b353b7931bce11926598773eb052bbe53506b903f2140a13ea1f54ead6e  libggml-cpu.so.0
```

To set the limits, add this top-level object to the controller dispatch JSON:

```json
{"trace":{"default_byte_cap":2097152,"max_byte_cap":16777216,"retain_jobs":8}}
```

These are the defaults. `GET /api/settings` shows the active values under
`trace`. Restart the controller to apply changes. The job cap includes all
manifests and tensor files for all prompts. It cannot exceed 33554432 bytes.
The cap does not limit model memory. Retention keeps 1 to 64 trace results
per worker. After a new result, the worker removes the oldest trace results
over the limit. It does not remove job records, inputs, or other artifacts.
An old job can show success after its trace files are removed. The trace
endpoints then return 404.

Submit one Library GGUF or HGN ID, or one scratch revision ID, in `input_ids` to
`POST /api/jobs` with `kind: "trace"`. Supply `prompt`, `prompt_id`, or an `options.prompts` array
with 1 to 4 entries. This example shows `options` for one prompt:

```json
{"layers":[0,11],"roles":["l_out","attn_out"],"positions":[0,1],"top_k":5}
```

Layers and token positions start at zero. The CLI does not use a chat
template or generate tokens. It accepts at most 4096 GGUF prompt tokens or
2048 HGN prompt tokens. HGN uses the raw tokenizer without special tokens. An
optional `byte_cap` can override the default within the configured maximum.
Set `full_dump: true` to pass larger explicit selections. This flag does not
select more data or raise the cap. Save the request to repeat the run.

For HGN only, set `options.hgn_projection` to a `direction_id`,
`source_layer`, and `target`. The controller loads version 2 of the direction
and verifies the checkpoint and its companion asset hashes. Select `residual`
to project all four streams, `dense` for the embedding and dense writers,
`experts` for routed expert output, or `all` for both writer groups. Use
inclusive `first_layer` and `last_layer`, or omit both to project all layers.
Capture `l_out`, `final_norm`, or logits. Do not combine this option with
GGUF interventions or SAE. This operation does not change model weights.

### HGN CPU setup

Use a regular worker with the Qwen3.8 Flash Next v2 `.hgn` checkpoint in its
Library. Add `"trace": true` to its dispatch capabilities. Dedicated Hy4
workers cannot run traces. For Frame, see the
[live provisioning handoff](deploy/lan/hgn-trace/LIVE-PROVISIONING.md).
Install the CPU environment from the same Brewery revision as the engine.
These commands use the network only for setup:

```sh
install -d -m 700 "$runtime"
uv venv "$runtime/venv" --managed-python --python 3.12.13
uv pip sync --python "$runtime/venv/bin/python" --only-binary :all: --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  tools/hgn_trace/requirements.lock.txt
cp --remove-destination "$(readlink -f "$runtime/venv/bin/python")" "$runtime/venv/bin/python"
sha256sum "$runtime/venv/bin/python"
```

Use absolute canonical paths and keep the environment and asset registry
under operator control. Set `BREWERY_HGN_TRACE_PYTHON` to the copied
`venv/bin/python`, `BREWERY_HGN_TRACE_PYTHON_SHA256` to its hash, and
`BREWERY_HGN_TRACE_ASSETS` to a JSON file with this layout:

```json
{
  "<checkpoint SHA-256>": {
    "sidecar": {"path": "/models/qwen38-flash-next-ngram.hgn", "sha256": "<sidecar SHA-256>"},
    "config": {"path": "/models/config.json", "sha256": "<config SHA-256>"},
    "tokenizer": {"path": "/models/tokenizer.json", "sha256": "<tokenizer SHA-256>"}
  }
}
```

Register the checkpoint itself as `hgn-v2` or `hgn`, not a sidecar or bundle
manifest.
The worker checks the model and all three assets before prefill. It checks the
runtime source pin and package versions and does not download files in a job.
Keep about 16 GiB of RAM free and run one large CPU trace at a time. Use a
narrow selection first. Each HGN `l_out` row has shape `[4,2560]` in
`[stream,hidden]` order. Capture `final_norm` at the last layer for the
normalized final mix. The lens uses the same mixer and head as model logits.
Other tensor roles are reported as missing. `hgn_projection` is the only
HGN intervention. Activation patches and SAE steering require GGUF.

Use job events to follow progress. After success, use
`GET /api/jobs/{id}/trace` for the summary. Use `/tensor` for F32 rows,
`/logit-lens` for layer logits, or `/model-logits` for native logits. All
four routes take the optional zero-based `prompt` index. Slice routes use
the half-open token position range `start` and `end`. Tensor reads also
require `layer` and `role`; logit lens reads can filter by `layer`. Narrow
the range if a response exceeds 4 MiB (HTTP 413).

### MoE routing atlas

Configure `BREWERY_LANTERNFISH_ROUTING` and
`BREWERY_LANTERNFISH_ROUTING_LIB_DIR` in the trace worker's `engine_env`.
Use the executable and four `.so.0` libraries from the CPU routing pin
(`lanternfish-x7-5699379`). Check `SHA256SUMS` before you install the files.
Keep the dense trace pin in place for other trace jobs. For a direct CLI
check, set `LD_LIBRARY_PATH` to the routing pin directory.

1. Start a Qwen2MoE or Qwen3MoE trace with all three roles:
   `ffn_moe_logits`, `ffn_moe_topk`, and `ffn_moe_route_weights`.
   Also capture `l_out` if you plan to compare an ablation. `top_k` is the
   number of model output logits. It does not set the number of experts.
2. Open the **MoE routing atlas** in Trace, call `routing_atlas`, or use
   `GET /api/jobs/{id}/trace/routing-atlas`. Select `layer`, `start`, `end`
   (exclusive), and optionally `prompt`. Read up to 128 contiguous
   captured positions from one layer per request.
3. To test an expert, run a second trace with the same model, prompt, and
   capture settings. Set
   `interventions: [{"layer":0,"role":"ffn_out","positions":[1],"op":"zero_expert","expert":3}]`
   for your layer, positions, and expert. Supply the second job ID as
   `ablation_job_id` on the baseline atlas. Only selected slots change.
   The comparison measures the layer's residual L2 change and, if both
   traces have model logits, shows whether the top token changed.

Load is the fraction of captured positions that select each expert.
Entropy is the mean entropy of the full router softmax, in nats.
Co-activation counts positions that select both experts. Specialization
groups selections by token ID and by the previous/current token ID pair.
The atlas shows up to five groups per expert, ranked by selection count.
These are observations of the selected prompts, not release qualification.

Pass `routing_evidence: {"job_id":"...","layer":0,"start":0,"end":4}` to
`plan_experts` to append source-bound routing evidence to its response.
The model ID and SHA-256 must match. This evidence does not replace the
independent REAP calibration and held-out inputs or change the REAP map.

## Tensor sensitivity

Use `sensitivity` to measure empirical Fisher for a dense Qwen2 or Llama F32
or BF16 GGUF. The job reads one Library GGUF or a scratch revision with a
private GGUF. It runs on the source worker. It does not change the model.

Set `"sensitivity": true` in the worker's controller capabilities. In the
worker's `engine_env`, set `BREWERY_LANTERNFISH_SENSITIVITY_PIN` to the
absolute path of the pinned runtime directory. Set
`BREWERY_LANTERNFISH_SENSITIVITY_MANIFEST_SHA256` to the trusted SHA-256 of
its `SHA256SUMS` file. The engine verifies the manifest and stages verified
copies of the CLI and the selected libraries from `lib/cpu`, `lib/cuda` or
`lib/rocm`. It uses the CPU libraries for tokenization. For example, the
Lanternfish pin `lanternfish-x4c-e0fd1b7` has manifest SHA-256
`9d82395374092c9794389c6833e0d8389230e87fc4e4c4c8c4de150082cef8dd`.
Use it for selected and corpus gradients. The selected-gradient finite
differences and CPU, CUDA, and ROCm parity checks passed. Scalar values and
gradient hashes matched the X3 pin for all 36 cases per objective.
Keep this runtime separate from the trace and fact-edit runtimes.

### X/A worker upgrade

Before you activate the merged worker, keep the W0 trace keys and paths.
Add the projection keys for `d842085`. Set the ablation binary key to the
`1bcb466` executable and use the same projection library path for both.
Do not change the routing, CPU edit, CUDA edit, or ROCm edit pins. Set the
sensitivity directory and manifest digest together to the X4c pin above.
If the worker serves HGN traces, configure its Python, interpreter digest,
and asset registry as shown in [HGN CPU setup](#hgn-cpu-setup).
Install the combined scratch-check runner, library path, hash, and matching
script together. Keep the prior worker configuration and binaries for
rollback. See [Checks before a publish request](#checks-before-a-publish-request).

For `cuda` or `rocm`, also set `BREWERY_ALLOW_GPU_LOAD` to `"1"` and
configure an `admission` wrapper that gets the GPU queue before it starts
the engine. For a private scratch GPU job, use the same local scratch GPU
configuration as for fact edits. GPU placement is strict. If the requested
backend cannot run the job, the job fails. BF16 can require a temporary F32
copy of the full model.

Submit `POST /api/jobs` with:

```json
{
  "kind": "sensitivity",
  "input_ids": ["<Library GGUF ID or scratch revision ID>"],
  "options": {
    "model_sha256": "<SHA-256 of the GGUF file>",
    "corpus_sha256": "9678f9afdc31b2a9be7fec2045997e2360693ee72b893287c0d539983bff4923",
    "backend": "cpu",
    "layers": [0],
    "observations": 4
  }
}
```

Select up to 16 `layers` (all two-dimensional weights in those layers), or
use `tensors` instead for up to 32 exact GGUF tensor names. The resolved
selection must fit 32 tensors, 256 gradient calls, 128 MiB for one gradient
and 2 GiB of total gradient data. Select 1 to 64 next-token observations in
the fixed W4 held-out corpus order (default 16). Start with one tensor and
a small observation count.

With MCP, use `tensor_sensitivity` with the same inputs and options but no
`kind`. Supply an `idempotency_key`, then read the result with `get_job`.
With REST, read `GET /api/jobs/{id}/sensitivity` after success. In Jobs,
open the job to see the heatmap and the exact numeric values.

The artifact records the GGUF and revision hashes, the corpus and token
observation hashes, and the runtime pin. Each tensor value is the mean of
squared next-token loss gradients over observations and tensor elements.
Each layer value is the mean over the selected elements of that layer.
The worker verifies each gradient, then removes the raw gradient data.
These values describe the selected observations only. They are
**not release-qualified**.

## Activation attribution

Use `find_circuit` to locate activation sites that can change a next-token
logit or logit margin. Use one dense Qwen2 or Llama F32 or BF16 GGUF
on its source worker. For a scratch revision, convert it to private GGUF first.
Enable `"circuit": true` in the controller
capabilities and configure the [Lanternfish runtime pin](#tensor-sensitivity).
The job uses `lib/cpu` only. BF16 can need a temporary F32 copy of the
model. The job does not change or publish model weights.

Submit `POST /api/jobs` with:

```json
{
  "kind": "circuit",
  "input_ids": ["<Library GGUF ID>"],
  "options": {
    "model_sha256": "<SHA-256 of the GGUF file>",
    "clean_prompt": "The capital of France is",
    "corrupt_prompt": "The capital of Germany is",
    "target_id": 12095,
    "contrast_id": 19846,
    "layers": [0, 8, 16, 23],
    "positions": [3, 4],
    "roles": ["l_out", "attn_out", "ffn_out"],
    "top_k": 8,
    "sample_lower": 4
  }
}
```

For MCP, omit `kind` and provide `idempotency_key`. Use `get_job` to read
the result. Use two distinct prompts with equal token counts from 1 to 128.
The CPU runtime tokenizes the raw text without a chat template. Layers and
positions start at zero. Omit `contrast_id` for a raw target logit. Use
at most 256 site gradients (two per site for a margin) and at most 32
patches in total. Set `sample_lower` (default 0) to also patch a reproducible
sample of sites below the top ranks. Start with a small selection.

The job multiplies each clean-minus-corrupt activation by the corrupt
gradient of the objective and sums the elements. It ranks sites by absolute
attribution. For each selected site, it makes a separate W2 trace that
replaces only that corrupt activation with the clean activation. A positive
measured change increases the objective from the unpatched corrupt value.
Pearson `r` and Spearman `rho` compare the signed attribution and measured
change for the patched sites only. Spearman uses mean ranks for ties.
A null value means that there were fewer than two measurements or that one
variable was constant. These correlations do not measure effects at
unpatched sites or interactions between sites.

Open the job in Jobs for the layer-by-position heatmap. Select a node role
and read the exact values and measured patches. Cite the saved
`GET /api/jobs/{id}/circuit` artifact by job ID, model SHA-256, and runtime pin.
The artifact includes both token sequences, baselines, and signed results.
The measurement is specific to this pair and is **not release-qualified**.

## Compare revisions and traces

### Quantization damage map

Enable `quantization_damage` in the worker's controller capabilities. Use the
same pinned Lanternfish configuration as for [tensor sensitivity](#tensor-sensitivity).
Register two single-file Library GGUFs from the same revision on that worker:
the dense BF16 or F32 reference first, then the quantized model. The job checks
their hashes, tensor names, and shapes. It does not prove that the two files
come from the same revision.

Submit `POST /api/jobs` with `kind: "quantization_damage"`, the two `input_ids`,
and `options` with `reference_sha256`, `quantized_sha256`, and
`gguf_type: "Q4_K_M"`. Do not supply a prompt. The default objective is
`target_logit`. It uses the first
`observations` lines of the pinned W4 held-out corpus (1 to 4, default 1).
By default it measures all quantized tensors. To reduce run time, set
`tensors` to exact GGUF names. Set `swaps` to 1 to 5 (default 3). CPU runs
both traces and swaps. With an admitted GPU, set `backend` to `cuda` or `rocm`
to run only the gradients on that backend.

Read `GET /api/jobs/{id}/quantization-damage` after success. With MCP, use
`quantization_damage_map` and `read_quantization_damage`. The artifact includes
W5 top-100 logit diffs. Each target is the highest-logit reference token at
the last prompt position. The job fails if a compared top-100 trace does not
contain that token. Positive drift means the reference logit is higher.
The job computes a first-order estimate of recovery: the F32 reference
gradient times (reference weight minus dequantized weight). If the quantizer
made a separate output head from tied embeddings, the gradient model uses
two copies of the reference weights for independent gradients. The job also
measures the change after it replaces each top-ranked tensor, one at a time,
in a temporary copy of the quantized GGUF. Predictions can differ from these
measurements. The job does not edit either input.

Set `objective: "corpus_nll"` to use mean teacher-forced next-token loss
over all 16 W4 held-out lines. Omit `observations`. The worker must use a
pinned Lanternfish build that supports `grad-corpus`. This objective does not
run traces or single-tensor swaps. `swaps` limits only the recipe exchanges.
The artifact gives the reference mean loss and observation count in `corpus`.
`predicted_nll_increase` is the dot product of the mean loss gradient and
the dequantized weight minus the reference weight. A positive value predicts
higher loss. The ranking uses absolute values. Verify the recipe with
measured losses; the gradient gives an estimate, not a measurement.

Review `recipe_proposal.layer_pairs`. Each pair exchanges tensor types between
equal-shaped matrices to preserve payload size. The proposal has no effect
until an operator saves `recipe_proposal.profile` with `POST /api/profiles`
and submits a separate `POST /api/conversions` with `format: "gguf"`, the
reference model ID, and the saved `profile_revision`. Use `uniform_profile`
to reproduce the unmodified Q4_K_M tensor assignments without implicit
row-width fallback. Both conversions use the same floating source and keep
its tensor inventory, including any tied output head. Compare actual sizes
and run W4 checks on both converted models before deciding whether to use
the proposal.
These results are **not release-qualified**.

Use these `GET` routes with `a=<baseline ID>&b=<comparison ID>`.
The routes do not change the inputs. A diff does not qualify a release.

- `/api/diff/tensors`: Use two scratch IDs from the same source, or one
  scratch ID and its original SafeTensors Library ID. Each file must be at
  most 4 GiB. The report lists changed tensors and counts elements with
  different stored bits. The bounds start at zero. The start is inclusive
  and the end is exclusive. The relative Frobenius norm uses the full
  baseline tensor as the denominator. A zero baseline gives `null`.
  For a matrix, the report shows up to eight singular values from a sample
  of at most 32 rows and 32 columns in the bounds. A sample can miss changes.
  Its rank counts sample values above one millionth of the largest value.
  It is not the rank of the full tensor.
- `/api/diff/traces`: Use two successful trace job IDs with the same prompt
  text, capture spec, and token IDs. Select one `prompt` index per request
  (default `0`). Use `layer`, `start`, or `end` to narrow the slice. The
  `layer` filter does not change the model logit rows. The report gives
  activation shifts and cosine similarity. It lists the ranked top-k for
  each side, plus token IDs that enter or leave. It calculates logit
  shifts and KL(A || B) only for shared top-k token IDs. This is not
  full-vocabulary KL. Without shared IDs, these values are `null`. Use
  worker and dispatch binaries that support trace diff reads.
- `/api/diff/behavior`: Use a scratch ID for `b` and its immediate parent
  for `a`. The controller reads `scratch/<b>/behavior.v1.json` under
  `BREWERY_DATA`. The file is at most 256 KiB and has at most 256 metrics.
  A producer must compare results from the same test inputs and settings.
  The producer writes the `BehaviorDiffInputV1` artifact from OpenAPI.
  Until it writes that file, the route returns 404.

Example behavior input (`parent_id` and `child_id` must be actual IDs):

```json
{"schema_version":1,"parent_id":"34868afe-7dbc-4f80-ba42-600481fd50e6","child_id":"0dcb24c4-bc87-4a7a-95d4-79c28b941446","metrics":[{"name":"locality.accuracy","parent":0.95,"child":0.93,"better":"higher","threshold":0.01}]}
```

Metric results can be numbers, Boolean values, strings, or `null`. A
numeric delta is the child value minus the parent value. A nonzero delta
with an absolute value at or below `threshold` gives `within_threshold`.
Other numeric changes give `improved` or `regressed` if `better` is
`higher` or `lower`.
The defaults are `threshold: 0` and `better: "neither"`. A missing value
gives `missing`. Other changes give `changed`.

The controller holds at most eight ordered reports in memory. It checks
trace retention before it returns a cached report. Report size is limited
to 4 MiB. Try a smaller trace slice on 413. Only one diff can run at a
time. On 429, wait and retry.

### Behavior directions

Use this CPU job to measure activation differences for `slop`, `sycophancy`,
or `response_length` on a Library Qwen2 GGUF or supported Qwen3.8 Flash Next
HGN v2 checkpoint (`hgn` or `hgn-v2`). The job does not change weights or
generate text.

1. Run `python3 tools/direction_prompt_cache.py /absolute/path/to/cache` on
   a machine with network access. The script checks the source size and
   SHA-256 before it reads any rows. Copy the generated digest-named `.json`
   files to a worker-owned directory. The worker can run offline.
2. In the worker `engine_env`, set `BREWERY_CONTRAST_CACHE` to that absolute
   directory. Configure the pinned Lanternfish runtime for GGUF or the pinned
   HGN Python runtime and model assets for HGN, as for a CPU trace. Enable
   `behavior_direction` on the worker that holds the model.
3. Send `POST /api/jobs` with `kind: "behavior_direction"`, one Library model
   ID in `input_ids`, and `options` with `behavior`, `model_sha256`, and
   `layers` in increasing order. Do not send prompts. The MCP tool
   `compute_direction` supplies the job kind.
4. Wait for the job to succeed. Read
   `GET /api/directions/{id}/versions/2` or MCP `get_direction` with the
   job ID and `version: 2`. This immutable artifact contains the model
   digest and the prompt and runtime pins. Each run makes a new artifact.
    Specify `layer` to read one layer instead of all vectors.
    For an older job, request version 1 to read its original raw-margin selection.

On GGUF, the default role is `l_out`. You can select `attn_out` or `ffn_out`.
The worker captures the prefix through the last user token and fits at that
token by default. On HGN, use `l_out` and select at most 25 layers for a
2560-wide model. The worker adds the assistant header with thinking disabled.
Set `last_k` to 1 to 16 to use that many final template tokens. If you do not
specify `last_k`, the job uses one token. For an explicit zero-based token
index below 512, set `position` instead. The HGN runtime accepts up to 2048
prompt tokens. At each position, it sums the four streams over their shared hidden
coordinates. The worker L2-normalizes that sum and averages the positions.
`mean_residual_norm` uses the norm before normalization.

To compare HGN results, pass `reference_id` and `reference_layer` to
`get_direction`. The reference must be a saved version 2 direction with the
same model and asset digests and reviewed behavior. The response gives the cosine for
each requested layer with a fitted vector, and null for other layers.
The comparison does not change either artifact.

The fit uses 12 prompt pairs and
scores 8 disjoint pairs. Each layer has the L2 norm of the training mean
difference, its unit vector, and the mean held-out signed projection of
positive minus negative rows (`held_out_margin`). The job divides this
margin by the pooled within-class sample standard deviation of the held-out
projections to get `held_out_cohens_d`. The largest positive score selects
the layer. Equal scores select the lower layer. Cohen's d is null if the
pooled standard deviation is zero or the direction has zero norm. These
layers cannot win. `mean_residual_norm` averages the `l_out` norms across both
classes of held-out prompts and all captured positions, even when the direction
uses another role on GGUF. Scores measure separation for these prompt pairs only.
They do not measure a change in generated behavior.
Results are not release-qualified.

For direction jobs, the cache script selects a small subset of each source
and pairs each original user text with two system instructions. It does not
vendor any rows:

- **Slop and response length:** Databricks, [Databricks Dolly 15K](
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
  Source SHA-256:
  `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.
  The derived sets use separate creative-writing and open-QA subsets.
  Share those derived sets under CC BY-SA 3.0 with this attribution.
- **Sycophancy:** Anthropic, [Language model evaluations: sycophancy on the
  NLP survey](https://github.com/anthropics/evals/tree/84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Source SHA-256:
  `582860b42e2beec806a7d361a08bcce7fdb264e2e697d55104074540352fb308`.
  The derived set keeps one persona per survey question and its agreement
  label. Keep this attribution if you redistribute it.

### Trace interventions

Use `options.interventions` for inline rules. Each rule specifies a `layer`,
`role`, and list of `positions`. Use `op: "add"` with a `vector` to add a
steering direction, `op: "replace"` with a `vector` to set the row,
`op: "scale"` with a `factor` to scale the row, or `op: "zero"` to set the
row to zero. The runtime applies rules before later nodes use the row.
Trace capture records the new values. Valid roles are `l_out`, `attn_out`,
`ffn_inp`, and `ffn_out`. A rule can target a row outside the capture
selection. Positions refer to the raw token sequence, not words.

To copy rows, set `options.activation_patches`. Each entry has
`source_job_id`, optional `source_prompt` (default `0`), `layer`, `role`,
and `positions`. For example:

```json
[{"source_job_id":"<trace-job-id>","source_prompt":0,"layer":1,"role":"l_out","positions":[0,1]}]
```

The controller reads the selected rows from a successful trace before it
queues the job. The source trace must still have the rows. Each source row
must be a verified, finite F32 vector. The controller copies each row to the
same layer, role, and position in every target prompt. It does not align
different token sequences. The source and target must have the same Library
or scratch model ID and GGUF SHA-256. The worker checks the target file before
the run.

You can combine inline rules, one saved artifact version, and up to eight
patch selectors. Do not target a row more than once. The combined limit
is 64 rules, with one rule per copied row, and 262144 vector values.
For HTTP 413 on a patch source read, split the selected positions into
smaller ranges. If retention removed a row, run the source trace again.

### Sparse autoencoders (SAEs)

Save an SAE with `POST /api/saes`. Send `multipart/form-data` with one
`metadata` JSON part and one `weights` SafeTensors part. A CLI client must
send `Origin` equal to the controller URL. Use an `Idempotency-Key` for
retries. The controller accepts at most 32 KiB of metadata and 128 MiB of
weights per version. It stores both in the controller database.

Bind the metadata to the Library GGUF or scratch `model_id`, the GGUF
SHA-256, a zero-based `layer`, and one vector `role`. The trace worker
checks the GGUF hash. Use `GET /api/saes` to list the latest versions.
Use `POST /api/saes/{id}/versions` to save a new version at the same binding.
Read metadata at `GET /api/saes/{id}/versions/{version}`. Append
`/safetensors` to download the weights. On HTTP 503, wait and retry.

Schema version 1 uses F32 tensors `W_enc[d_in,d_sae]`,
`W_dec[d_sae,d_in]`, `b_enc[d_sae]`, and `b_dec[d_in]`. For `jump_relu`,
also include a nonnegative `threshold[d_sae]`. The encoder scales the
input by `normalization.scale`, optionally subtracts `b_dec`, then applies
the encoder weights and bias. `relu` keeps positive values. `top_k`
keeps up to `k` positive values. `jump_relu` keeps values strictly above
each feature threshold. Record the source or training method in
`provenance`. Do not use pickle files.

To read features, set `options.sae` to `{"id":"<sae-id>","version":1,"top_k":5}`
in a trace request. Capture the SAE layer and role. After the job
succeeds, use `GET /api/jobs/{id}/trace/features?prompt=0&start=0&end=16`
or MCP `read_feature_trace`. Select 1 to 16 token positions per read.
In the Trace view, select a token position under **Top SAE features**.
Feature reads depend on the same retention settings as tensor reads.

To steer a feature, use MCP `steer_feature` or add
`options.feature_steering` to a trace request:

```json
{"id":"<sae-id>","version":1,"feature":0,"alpha":8.0,"positions":[3]}
```

The trace adds `alpha` times the stored decoder row at each selected
position. If the SAE uses a scale other than 1, choose `alpha` in units of
the stored decoder direction. This rule must not overlap another
intervention. This operation does not change model weights. Compare it
with a trace of the same prompts and positions without steering. Results
are **not release-qualified**.

Use [the CPU SAE procedure](qualification/weight_space_sae.md) to train a
small TopK SAE from trace rows or to load the pinned public ReLU example.

### Crosscoder model diff

Use a crosscoder to compare two traceable GGUF revisions at one layer and
vector role. Use the same prompt text, token positions, layer, and role on
both revisions. Use different prompts for training and held-out tests. Do
not use interventions. Finish the trace reads before the worker removes
the trace files.

This method follows the shared features and model-specific decoders in
Anthropic's 2024 [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html).
The trainer uses W11 TopK instead of the paper's sparsity penalty. It
uses one code for both sides and normalizes each pair of decoder
directions together. It uses the W11 seed and Adam settings and two CPU
threads. `--difference-weight` puts more weight on the
reconstruction error in the child-minus-parent direction. Its default
value of 1 uses ordinary reconstruction loss.

With the CPU PyTorch environment from the
[CPU SAE procedure](qualification/weight_space_sae.md), make a spec:

```json
{
  "layer": 8,
  "role": "l_out",
  "pairs": [
    {"parent_job_id":"<train-parent-job>","child_job_id":"<train-child-job>","split":"train"},
    {"parent_job_id":"<test-parent-job>","child_job_id":"<test-child-job>","split":"heldout"}
  ]
}
```

Run the trainer, then import the two output files. Set `base` to the
controller URL and use a new path for `out`.

```sh
python tools/train_crosscoder.py "$base" paired-traces.json "$out"
curl --fail-with-body -H "Origin: $base" \
  -F "metadata=@$out/metadata.json;type=application/json" \
  -F "weights=@$out/crosscoder.safetensors;type=application/octet-stream" \
  "$base/api/crosscoders"
```

The controller checks both bindings against W1 job receipts. Each binding
has the trace input ID and exact GGUF SHA-256. The report also records
the scratch weight hash, if present, and the trace manifest hashes in
pair and prompt order. An exact retry returns the
same immutable version 1 artifact. Read it at `GET /api/crosscoders/{id}`,
with MCP `crosscoder_diff`, or in **Crosscoder** in the workbench. Append
`/safetensors` to the REST path to download the weights.

The encoder flattens parent then child and uses `W_enc[2*width,features]`
and `b_enc[features]`. It multiplies rows by `scale`, subtracts
`b_dec[2,width]`, applies the encoder, then keeps at most `k` positive
values. Each side reconstructs with its own `W_dec[2,features,width]`
slice and bias, then divides by `scale`.

Read the parent and child held-out MSE, explained variance
(`1 - MSE / centered variance`), and mean active feature count (L0).
A side counts a feature only if its decoder
norm exceeds `1e-8`. Dead counts use this rule and training activity.
The child norm ratio is the child norm divided by the sum of both norms.
A live feature is parent-only at a ratio of 0.10 or less, or child-only
at 0.90 or more. Other live features are changed if the decoder cosine
is less than 0.995. These thresholds describe the learned dictionary.
They do not prove that a feature caused a model change. Reconstruction
metrics come from the trainer and are not recomputed during import.
Results are **not release-qualified**.

### Library intervention artifacts

Use `POST /api/interventions` to save an artifact. Give it a name, the
Library GGUF or scratch `model_id`, the GGUF SHA-256, and the inline rules:

```json
{
  "name":"Residual direction",
  "model_id":"<library-model-id>",
  "model_sha256":"<64 lowercase hex digits>",
  "interventions":[{"layer":1,"role":"l_out","positions":[0],"op":"add","vector":[0.25,-0.5]}]
}
```

Use one vector value per hidden dimension. The two values above show the
format only. `GET /api/interventions` lists the artifact IDs and latest
versions. `GET /api/interventions/{id}/versions/{version}` reads the rules
for one version. To save new rules, send `{"interventions":[...]}` to
`POST /api/interventions/{id}/versions`. Versions start at 1. A new version
does not change an earlier version or its model binding. Use an
`Idempotency-Key` to retry a write without creating another version.

In a trace request, set `options.intervention_artifact` to
`{"id":"<artifact-id>","version":1}`. The controller rejects a different
model ID. If the registered file has a different SHA-256, the worker
rejects it before it runs Lanternfish. A saved artifact does not change
model weights or qualify the model for release.

#### Direction ablation and subtraction

For projection and subtraction traces, set `BREWERY_LANTERNFISH_PROJECTION`
to the pinned `d842085` binary and `BREWERY_LANTERNFISH_PROJECTION_LIB_DIR`
to its library directory. The binary SHA-256 is
`3009f4b0c362ddc04d955e40efac9e1d88659b9a0bc393fdc8ea6da287bde010`.
Keep the W0 settings for plain traces and W2 rules. Do not combine
projection or subtraction with MoE routing in one trace.

The projection runtime and the ablation screen use the same four libraries:

```text
f4a59a07c8ebea5d02779f82e16cf23d5815a288fede9cdbf851f73b1854c005  libllama.so.0
c51f659bc7c64ab3b675fb56cdddf1beab54c75818ace8325c7e49ff1a60e3e4  libggml.so.0
c445afaddb3d37bbb8ef6d979786f66f4cae71b5597fee0feb398f9b961289b8  libggml-base.so.0
fd7b911f677b93bef0b2fd31877da45bc5e78bd7197fe33df5ed4b76f75b08cd  libggml-cpu.so.0
```

To use an A1 behavior direction, send `direction_rules` in place of or with
`interventions`. Each rule names the immutable direction ID and version, the
layer to read from that artifact, and the target layer, role, and positions.
The direction must match the model ID and GGUF SHA-256. The controller saves
the reference and the resolved unit vector in the intervention version.

```json
{
  "direction_rules":[{
    "direction_id":"<direction-job-id>","direction_version":2,"source_layer":15,
    "layer":15,"role":"l_out","positions":[4],"op":"project_out"
  }]
}
```

Use `project_out` to remove the activation component along the direction.
Use `subtract` with a finite, nonnegative `scale` to subtract a fixed multiple
of the unit vector. Save these rules with `create_direction_intervention`
or send the rules to either intervention write endpoint. Use the artifact
with `start_trace`. Compare the edited activation and logits with an
unmodified trace of the same prompt and capture spec.

Use `GET /api/interventions/{id}/versions/{version}/gguf` to download a
GGUF v3 control vector. Pass the file to llama.cpp with `--control-vector`.
Export requires one `add` or `subtract` rule per `l_out` layer and the same
vector width for all layers. A `subtract` rule exports the negative scaled
vector. llama.cpp control vectors cannot express a projection because the
removed component depends on the current activation. The llama.cpp adapter
does not use layer 0. For layer `N`, the F32 tensor name is `direction.N`. The file
has `general.architecture: "controlvector"`, `controlvector.model_hint`
with the Library model name, and `controlvector.layer_count` with the
number of tensors. GGUF does not store token positions or enforce the
Brewery model binding. Use the bound model. llama.cpp applies the
direction to all tokens at that layer.

## Operator paths and environment

The operator sets these variables. The controller uses the first group. The
worker environment applies to jobs that run on that worker.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `BREWERY_LISTEN` | controller | Listen address. Default `127.0.0.1:4141`. |
| `BREWERY_DATA` | controller | Absolute path of the controller state directory. |
| `BREWERY_WEB_DIST` | controller | Built web page. Default `web/dist`. |
| `BREWERY_DISPATCH` | controller | Path of `brewery-dispatch`. Default: the one on `PATH`. |
| `BREWERY_DISPATCH_CONFIG` | controller | Absolute path of the dispatch JSON file. |
| `BREWERY_WORKER_CONFIG` | worker | Worker JSON path. Default `~/.config/model-brewery/worker.json`. |
| `BREWERY_CPU_THREADS` | worker, engine, native tools | CPU work budget per process. Default `4`; range `1`–`64`. Set this on each worker host. Available CPUs and the number of independent work items can lower the effective count. |
| `BREWERY_HASH_THREADS` | worker, engine | Optional limit for concurrent file hashes (`1`–`16`). Hashing uses the CPU budget by default, up to `16` files at a time. |
| `BREWERY_LANTERNFISH` | engine | Configured Lanternfish `run` binary for `compare`. This lookup does not check the trace pin. |
| `BREWERY_LANTERNFISH_TRACE` | engine | Absolute path to the pinned CPU Lanternfish binary for `trace`. |
| `BREWERY_LANTERNFISH_TRACE_LIB_DIR` | engine | Absolute path to the directory with the four pinned CPU trace libraries. |
| `BREWERY_LANTERNFISH_ROUTING` | engine | Absolute path to the pinned CPU Lanternfish executable for MoE routing and expert ablation. |
| `BREWERY_LANTERNFISH_ROUTING_LIB_DIR` | engine | Absolute path to the matching directory with the four pinned routing trace libraries. |
| `BREWERY_HGN_TRACE_PYTHON` | engine | Absolute path to the copied CPython 3.12.13 HGN trace environment executable. |
| `BREWERY_HGN_TRACE_PYTHON_SHA256` | engine | SHA-256 pin for that executable. |
| `BREWERY_HGN_TRACE_ASSETS` | engine | Absolute path to the HGN model asset registry keyed by checkpoint SHA-256. |
| `BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR` | engine | Absolute path to the directory with the four pinned CPU scratch-check libraries. |
| `BREWERY_LANTERNFISH_PROJECTION` | engine | Absolute path to the pinned CPU executable for projection and subtraction traces. |
| `BREWERY_LANTERNFISH_PROJECTION_LIB_DIR` | engine | Absolute path to the pinned CPU libraries for projection traces and ablation screens. |
| `BREWERY_LANTERNFISH_ABLATION` | engine | Absolute path to the pinned CPU executable for ablation screens. |
| `BREWERY_LANTERNFISH_EDIT` | engine | Absolute path to the pinned CPU Lanternfish binary for locate-then-edit jobs. |
| `BREWERY_LANTERNFISH_EDIT_LIB_DIR` | engine | Absolute path to the directory with the four pinned llama.cpp and ggml libraries. |
| `BREWERY_LANTERNFISH_EDIT_CUDA` | engine | Absolute path to the pinned W14b Lanternfish binary for CUDA locate-then-edit jobs. |
| `BREWERY_LANTERNFISH_EDIT_CUDA_LIB_DIR` | engine | Absolute path to the directory with the five pinned W14b libraries. |
| `BREWERY_LANTERNFISH_EDIT_ROCM` | engine | Absolute path to the pinned Lanternfish binary for ROCm locate-then-edit jobs. |
| `BREWERY_LANTERNFISH_EDIT_ROCM_LIB_DIR` | engine | Absolute path to the directory with the pinned ROCm edit runtime libraries. |
| `BREWERY_CONVERTER` | engine | Pinned bf16-to-hgn converter for `inspect`, `plan`, `convert`, and `verify`. |
| `BREWERY_COMPONENT_CONVERTER` | engine | Separate role-overlay converter build for HF jobs that replace a component. See [Conversion components](#conversion-components). |
| `BREWERY_COMPONENT_CPU_ONLY` | engine | Set to `1` to refuse non-CPU backends for those jobs. |
| `BREWERY_CUSTOM_CONVERTER` | engine | Separate CPU-only converter for custom profiles. See [Custom profiles](#custom-profiles). |
| `BREWERY_GGUF_QUANTIZER` | engine | CPU-only `llama-quantize` for GGUF output. See [GGUF output](#gguf-output). |
| `BREWERY_GGUF_HF_EXPORTER` | engine | Optional bounded HF-to-GGUF exporter. |
| `BREWERY_EVAL` | engine | Path of `brewery-eval`. |
| `BREWERY_ALLOW_GPU_LOAD` | eval | Set to `1` to let `generate` and `evaluate` start an isolated Halogen load. Without it, those jobs fail. |
| `BREWERY_HALOGEN_IMAGE_ID` | eval | Full local image ID, `sha256:` and 64 hex digits. Get it from `docker image inspect`. A tag is not accepted. |
| `BREWERY_BASELINE_CONFIG` | eval | Absolute path of the reference endpoint JSON file. |
| `BREWERY_BASELINE_TOKEN_*` | eval | Bearer token for a configured reference endpoint. |

Each file retains its ordinary SHA-256 checksum. A single file uses one
reader. Set the thread limit in the worker process environment on each
host, not in a browser job. Use `1` for serial reads on disks where
parallel reads are slower. Existing pinned binaries do not gain this
behavior until upgraded.

For example, set `BREWERY_CPU_THREADS=8` in the worker service environment
to allow eight CPU threads where the job has independent work. `engine_env`
alone does not configure the worker's own hashing or transfer processes.
The engine passes the effective budget to local native tools and their
OpenMP and numerical libraries. This setting does not configure a remote
inference endpoint. It also does not run dependent tokens out of order,
skip GPU admission, or allow multiple publishers for the same artifact.

For SSH workers, set native executable paths in `engine_env` in the worker
configuration instead of relying on an interactive shell's `PATH`. Enable
`baseline`, `generate`, `evaluate` or `benchmark` in the controller host's
`capabilities` only when that worker has the corresponding runtime. Use the
[evaluator setup](crates/eval/README.md) for the baseline file format and
endpoint identity contract. Use [Managed benchmarks](#managed-benchmarks) for
`BREWERY_BENCHMARK_CONFIG` and the `benchmark` job.

### Baseline endpoints

Use `options.baseline_source` to compare with a configured live endpoint.
The endpoint can serve a different source or precision from the candidate.
Its `source_sha256` identifies the reference model, not the candidate.
The endpoint file maps an ID to an entry:

```json
{
  "int8-reference": {
    "url": "https://reference.example/v1/completions",
    "model": "model-name-as-the-endpoint-reports-it",
    "source_sha256": "<64 lowercase hex digits of the reference source>",
    "runtime_id": "runtime-1",
    "precision": "int8",
    "bearer_env": "BREWERY_BASELINE_TOKEN_REFERENCE"
  }
}
```

The URL must use `https`, or `http` on `127.0.0.1` or `[::1]`. The path must
be exactly `/v1/completions`. On each row, the endpoint must return
`X-Brewery-Source-Sha256` and `X-Brewery-Runtime-Id` headers that match this
entry. Its response `model` must also match the entry. The evaluator follows
no redirects and uses no proxy.

These checks compare the endpoint's replies with operator configuration.
They do not verify the served weights or precision. The optional `precision`
value is a descriptive label. Ordinary comparisons do not require the same
source or precision as the candidate.

Worker **Settings** can save optional `precision`, but cannot mark a reference
as BF16-attested. A trusted evaluator endpoint configuration can set
`bf16_attested: true` only with `precision: "bf16"` and a non-synthetic
endpoint. The evaluator reports `attested_bf16_same_source` only when this
endpoint's source hash also matches the candidate. This operator assertion
does not replace the separate BF16 release checks below.

## Job flow

In **Library**, choose a configured host before entering a model path.
Start typing in **Path on host** to look up matching directories and files
on that worker, including over SSH. An empty prefix offers the worker's
configured source roots. A suggestion does not register a model, read its
contents, or queue a job. If the worker cannot be reached, you can still
enter an absolute path. The worker checks it when a job uses it.

1. Register a model. Give a host, an absolute path on that host, and a kind.
   The workbench does not guess size or hash at this step. The worker checks
   them when a job uses the file.
2. Save a prompt in the library.
3. Create a job. For a conversion, use the one-submit workflow in
   [Convert a model](#convert-a-model). Pick a worker or select **Any
   eligible worker**. Jobs in the configured conversion pool can run
   concurrently, one per worker. Other workflows wait for the compute queue
   to drain.
4. If an input is on a different host, the controller queues a transfer. Use
   `transfer_preview` first to see missing bytes and free space. A transfer
   resumes after an interruption. The worker checks hashes and reuses a verified
   copy.
5. Watch progress in the web page. You can cancel a job. After a browser
   reload, the page reconnects to the job.
6. Read the result. `GET /api/jobs/{id}` returns the full result and the
   controller event log.

For jobs that the workbench can submit, open the job's view and enter the
inputs again. To repeat a failed `hf_download` on a named remote worker,
submit it through `POST /api/jobs` with the same `host`. The Library form
uses the controller's download service and cannot select a remote worker.

If you lose a reply after you submit a job, check **Jobs** first. If you
submit again, use the same inputs and do not leave or reload the page.

### Convert a model

The default way to convert is one action. On the Convert page, choose the
source, the profile and the worker, then submit once. Agents use
`POST /api/conversions`. The controller then runs four jobs in this order:
`inspect`, `plan`, `convert` and `verify`. The controller stores the
workflow. It keeps running if you close the browser or the client
disconnects. After a controller restart, it continues from the stored job IDs.
It does not retry a running job that it cannot check.

- The workflow state is `running`, `succeeded`, `failed` or `cancelled`.
  `running` includes the time that a job waits in the queue. The `stage` field
  names the current stage.
- The first failed or cancelled job ends the workflow. Later stages do not
  start. The workflow `error` repeats the error of the failed job when the
  worker gave one. Read each job with `GET /api/jobs/{id}` for its output.
- Cancel the whole workflow with `POST /api/conversions/{id}/cancel`. The
  controller cancels the active job and starts no later stage. A running job
  needs time to stop. Poll until the state is terminal. A timeout is not proof
  that the worker stopped.
- A model record for the converted output can exist before `verify` passes.
  Check the workflow `state`, not only `artifact_model_id`.
- Each stage writes its own record. A `succeeded` workflow means that all four
  jobs passed. It does not show generation quality, GSM8K quality or release
  qualification. To check quality, run `generate` and `evaluate`.
  Evaluation scores the candidate without a reference by default.
  Add an optional reference to compare scores. A comparison is not a release
  qualification.

The individual `inspect`, `plan`, `convert` and `verify` jobs remain for
diagnosis. Use them to run or repeat one stage. Do not use them for a normal
conversion. A passed file check does not show that generation or benchmark
quality is correct.

Completed model bundles stay on the host that made them. The controller gets
a receipt. To move a whole bundle, run a `transfer` job.

### Conversion components

A conversion uses four components besides the main weights: tokenizer and
config files (`assets`), the vision tower (`vision`), the engram lookup table
(`ngram`) and the MTP draft head (`mtp`). The converter accepts only the
Qwen3.8 Flash-Next layout. The converter documentation has its full rules.

**Default: no component input.** Register only the source and submit.

- **HF source:** The converter takes all four components from the snapshot.
  Vision is `model.visual.*`, the lookup table is the 128 numbered n-gram
  shards, and MTP is `mtp.*`. It finds them through the fixed index of the
  snapshot. The assets are the flat tokenizer and config files.
- **GGUF source:** The converter needs matching `assets`, `mtp` and `vision`
  inputs. It needs `ngram` only if the GGUF has no embedded table.
- **HGN source:** The converter uses the components that the bundle holds.
  It needs an input for each missing component.

On the Convert page, select a source checkpoint to see its component choices.
The checkpoint is the default; you do not need to register its embedded
components separately. The form uses saved job reports to identify components.
If no report confirms a role, the form does not claim it is present. To
override a role, select a separate Library entry. After conversion, read
`result.components` to see what the worker reports using.

**Replace a component of an HF source.** This works only on a host that
reports `capabilities.convert_components: true` in `GET /api/state`. The
value is `false` unless the operator turns it on. If a host reports `false`,
do not submit a replacement for an HF source to it. The `reasons` field of the
host can say why. Each role accepts one input:

| Role | Accepted input |
| --- | --- |
| `assets` | A flat directory of tokenizer and config files. |
| `vision`, `ngram`, `mtp` | A role-only SafeTensors file, a flat directory that holds exactly one SafeTensors index and its shards, or a native `.hgn` file with the matching role identity. |

The rules that the native reader applies:

- An explicit role replaces the whole embedded role. A role that you do not
  supply stays embedded. The full HF snapshot is still required. The index
  always comes from the snapshot.
- Vision uses `model.visual.*`. MTP uses `mtp.*`. The lookup table uses only
  the numbered n-gram shards 0 to 127. The heads, vocabulary and multipliers
  stay with the snapshot.
- A replacement must have exactly the expected tensor names, shapes and a
  compatible data type. Missing or extra tensors fail the job.
- Assets must be regular, flat files from the allowed set. The set must equal
  the set of the snapshot. Config files must be byte-identical. There is no
  fallback for single files.
- Symbolic links and hard links as role aliases are rejected. One input cannot
  overlap the root of another input.

**Register a path, not bytes.** A component is a normal Library entry with
the kind `assets`, `vision`, `ngram` or `mtp`. Registration stores a host and
an absolute path. It does not upload, hash or check the files. The controller
rejects a path that is not absolute, or that has `..` or a control character.
The worker and the engine check the format when a job uses the entry. Put the
component on the host that runs the job, inside a configured source root, or
transfer it first.

**Select by index.** Add the component IDs to `input_ids` after the source.
Set `options.assets`, `options.vision`, `options.ngram` or `options.mtp` to
the zero-based index of that ID. Each kind appears at most once, and each
index must point to an input of that kind. `inspect`, `plan` and `convert` use
the source at index 0. For `verify`, index 1 is the HGN output, so component
indexes start at 2. The one-submit workflow takes the same `input_ids` and
`options` as `convert` and passes them to each stage. The workflow compares
the `source_sha256` of the later stages with that of `inspect`. If it differs,
the workflow fails, because the snapshot or a selected component changed after
inspection.

`hf` has the alias `bf16-checkpoint`. The controller stores the kind `hf`.
The component kinds have no alias.

**Isolation.** A component replacement does not run on the standard
converter. The operator sets up a separate worker host with its own converter
build:

| Setting | Where | Meaning |
| --- | --- | --- |
| `capabilities.convert_components` | dispatch host entry | `true` turns on component replacement for HF sources on this host. Default `false`. |
| `component_cpu_only` | dispatch host entry | If `true`, the controller rejects every backend option other than `cpu`, including the workflow `decode_backend`. The state shows it as `component_cpu_only`. Default `false`. |
| `BREWERY_COMPONENT_CONVERTER` | engine | Path of the separate role-overlay converter build. |
| `BREWERY_COMPONENT_CPU_ONLY` | engine | Set to `1` to make the engine refuse non-CPU backends on its own. |

Jobs without a replacement keep the standard converter and its checks. The
approved source digest and the GPU admission checks stay as they are. The API
has no route to change these settings.

**What the result proves.** `result.components` names, for each role, the
source: `input`, `embedded`, `none` or `unknown`. For an `input`, it adds
`input_id`, `format` (`assets-directory`, `safetensors`, `safetensors-index`
or `hgn`), `sha256`, `bundle_sha256` and `bytes`. The native report names the
mode `hf-role-overlays-v1`. The converted manifest records
`source_provenance.format` as `huggingface-composite-v1`. It binds the
snapshot, the replaced roles and the selected bytes. Results from older
workers have no `components` field. Read that as "not reported".

`verify` detects a change of the selected tensor bytes. It checks the payload
integrity and the decoded values. It does not show that a replacement comes
from the same checkpoint as the snapshot. It does not show model quality. The
caller must know where each replacement came from. A passed `verify` is not a
release qualification.

### Custom profiles

A built-in profile (`v2` or `ht43`) gives each tensor one encoding. A
**custom profile** starts from one of them and moves chosen tensors to the
encoding of the other. It cannot make a new encoding, and it cannot change a
tensor that the catalog marks as not editable. The tensors that the editor can
change are the hyper-connection and routed-expert tensors of the trunk, and
the hyper-connection tensors of the global group. The embedded catalog
(`crates/controller/src/profile_catalog.json`) covers one architecture,
`qwen4_exp`. It lists every tensor with its two encodings and its signed byte
difference.

**Edit.** The **Profiles** page (opened from **Convert**) is the editor. It shows the trunk layers
and the other groups from the catalog. You can set a tensor, a layer, a role,
or a group to the alternate encoding. The editor sends only the list of
changed tensors. Saving a profile starts no job.

**Revision.** `POST /api/profiles` saves a profile. The controller checks the
catalog digest, the name (1 to 256 bytes, no control characters), the number
of changes (1 to 4096) and that each change names an editable tensor with the
other encoding. The revision is the SHA-256 of the canonical recipe. A profile
never changes after saving. An edit is a new revision. The same content
returns the same revision (`200`; a new revision is `201`). `GET /api/profiles`
lists all revisions, and `GET /api/profiles/catalog` returns the catalog. The
body of a save is at most 512 KiB.

**Use.** Add `profile_revision` to `POST /api/conversions`. Do not also set
`options.profile`. The request is refused if the source is not `hf`, if a
backend is not `cpu`, or if the host does not report
`capabilities.custom_profiles: true` (`409`). An unknown revision returns
`404`. The workflow runs all four stages with the saved content. The request
cannot change it. The workflow and its child jobs show `profile_revision`.
The recipe is not part of the job options or of the job detail.

Three statements have different strength. Do not mix them.

| Statement | Basis | Strength |
| --- | --- | --- |
| **Predicted size** (`size.predicted_bytes`) | The base size of the catalog plus the signed deltas of the changed tensors. The deltas come from the layout of the catalog, with the companion tensors that change with each choice. | A layout estimate. It is not a measured output. |
| **Custom file check** (`verify` of the workflow) | The native tool checks that the recipe and the output match: the plan, the manifest and the main container. It also checks source provenance, payload integrity and decoded values. | Proves that the file was made as the saved recipe says. |
| **Runtime accuracy and quality** | Not measured by saving, converting or verifying. | **Unproven.** Run `generate` and `evaluate`. Evaluation scores the candidate by default. Add a reference only to compare scores. A comparison is not a release qualification. |

The engine writes `custom_profile` in the result with the name, the revision,
the catalog digest, the architecture, the base, the recipe digest and
`release_qualified: false`. The chosen encoding of each tensor is in the native
plan and in the manifest. The output has no standard runtime manifest.

**Isolation.** A custom profile runs only on a worker that the operator set up
for it:

| Setting | Where | Meaning |
| --- | --- | --- |
| `capabilities.custom_profiles` | dispatch host entry | `true` turns on custom profiles on this host. Default `false`. |
| `BREWERY_CUSTOM_CONVERTER` | engine | Absolute path of a separate converter build. It must differ from the standard converter and run on the CPU only. |

The engine gives the recipe to the converter as a private temporary file
(`--recipe`) in the job directory, outside the output. A worker that pins a
source for release qualification refuses custom profiles. The standard
converter and the pinned qualification jobs do not change.

### GGUF output

A conversion can write an HGN bundle or a GGUF file. The two paths are
separate. HGN is the default and it did not change.

**Select the format.** Add `"format": "gguf"` to `POST /api/conversions`. If
`format` is missing, or it is `hgn`, the HGN workflow runs as before. The
same four stages run for GGUF: `inspect`, `plan`, `convert` and `verify`. A
saved profile has a format too, and a conversion accepts only a profile of its
own format.

**Exact export (development only).** The target is one quantization policy
with a separate choice of HGN or GGUF. Changing the container must preserve
the resolved tensor encodings, scales, rotation data and other model assets.
Use a stock representation when it is exact. Otherwise use the separately
versioned `brewery.cross-container.v1` reader, not a different quantization.
A missing AMD reader is not evidence that an encoding requires NVIDIA.

`tools/exact-export --encoded-sha256 DIGEST --to hgn|gguf SOURCE DEST`
operates on one already verified encoded file of at most 64 MiB. Use the
digest from the verified source. For the same format, the tool copies the
file and checks the output digest. For a different format, it requires a
fixture adapter that can pack and verify the input. It does not run a
quantizer or overwrite a destination. This tool does not export external
components or establish that a model loads, runs on AMD, or retains its
quality. It is not full HGN-to-GGUF support; `hgn` remains refused as a GGUF
source below. It is not a production worker or a release qualification step.

**Inputs.** `input_ids` is the source, then at most one `imatrix` input.

| Source | Rule |
| --- | --- |
| `gguf` | One file. It must hold only F32, F16 or BF16 weights (and integer tensors). A quantized or split GGUF is refused. The host needs `capabilities.gguf_output`. |
| `hf` | The host needs `gguf_hf_output` as well. The exporter accepts only bounded Qwen2 and GPT-2 model forms with exact tensor names and shapes. See the limits below. |
| `safetensors` | Set `input_ids` to `[weights, assets, optional imatrix]`. Register `weights` as one standalone `.safetensors` file and `assets` as a `kind=assets` directory. Keep all inputs on one worker. Include `config.json`, `tokenizer.json` and `tokenizer_config.json` in `assets`. The worker does not read adjacent files. The host also needs `gguf_safetensors_output`. |
| `hgn` | Refused. |

**Presets.** `options.preset` is one name from the GGUF catalog
(`GET /api/profiles/gguf-catalog`). The default is `Q4_K_M`. The only other
GGUF option is `threads` (1 to 64). A preset is a whole-model recipe of the
pinned upstream quantizer. The quantizer chooses the type of each tensor.
`decode_backend` and all HGN options are refused. The catalog is pinned to one
encoder revision. It lists only types that the encoder can write. It has no
`IQ1_XXS`, and it has no F32 preset. A name that is not in the catalog is
refused.

**ROCmFPX presets and tensor types.** The catalog also lists five presets from
a separate encoder. It pins `charlie12345/ROCmFPX` to commit
`fb08d7cdb67082ddda15a25e64dc7ec413afc41c`. The table separates each GGUF
`general.file_type` value from its per-tensor GGML type ID.

| Preset | `general.file_type` | Fork tensor type and GGML ID | Bytes per 32 weights |
| --- | ---: | --- | ---: |
| `Q3_0_ROCMFPX` | 112 | `Q3_0_ROCMFPX` (104) | 14 |
| `Q4_0_ROCMFP4` | 100 | `Q4_0_ROCMFP4` (100) | 18 |
| `Q4_0_ROCMFP4_FAST` | 103 | `Q4_0_ROCMFP4_FAST` (101) | 17 |
| `Q6_0_ROCMFPX` | 110 | `Q6_0_ROCMFPX` (102) | 26 |
| `Q8_0_ROCMFPX` | 111 | `Q8_0_ROCMFPX` (103) | 33 |

These five fork types are the only ROCmFPX tensor overrides. Each override
must use one exact tensor name and one exact type from the catalog. The engine
escapes the tensor name and anchors the match before it calls the quantizer.
No ROCmFPX type requires an imatrix. A ROCmFPX selection must emit at least
one fork tensor type. The engine refuses a dry run that emits none.

A stock preset, such as `Q8_0`, can carry an exact ROCmFPX tensor override.
That selection uses the fork encoder. The pinned fork does not
define stock `Q2_0` (GGML type ID 42). The catalog excludes `Q2_0` from
ROCmFPX selections. Other stock presets and tensor types remain available.

**Precision and mixed presets are separate things.** A *precision* preset
(`F16`, `BF16`) stores every weight in one floating-point type. It takes no
tensor override. A *mixed* preset (for example `Q4_K_M`) lets the quantizer
use several types. You can then save a **GGUF profile** with
`POST /api/profiles` and `"format": "gguf"`. It holds a `preset` and up to 256
`overrides`. An override maps one exact tensor name (at most 63 ASCII
characters) to a raw tensor type from the catalog. The controller checks the
type names when you save. It cannot check the tensor names then, because it
has no inventory. The engine checks them when the conversion runs. A GGUF
profile has no size estimate. Use `profile_revision` or `options.preset`, not
both.

In **Profiles → GGUF**, select a Library model to load its tensor names.
You do not need to select an inspect job. The editor uses exact GGUF names
when a saved report or a matching header plan supplies them. For Flash-Next,
the plan lists the main GGUF tensors, not the separate vision projector or
metadata arrays. A name in the plan does not mean conversion has passed.
When only the source inventory is available, its unmapped names are read-only.

**Imatrix.** An importance matrix is a Library entry of the kind `imatrix`.
You register a path on the worker, as for other inputs. A low-bit preset or
type needs it: in the catalog these have `requires_imatrix: true` (for
example the `IQ1_*` and `IQ2_*` presets, `IQ3_XXS`, `IQ3_XS` and `Q2_K_S`).
Without the input, the controller returns `400` and the engine also refuses.
The file must be a regular file of 1 byte to 256 MiB. The controller checks
only the path. The worker checks the file when a job uses it. The imatrix does
not prove that the quantization is good.

**Prerequisites.** The operator sets up the GGUF path on a separate worker. It
is off by default.

| Setting | Where | Meaning |
| --- | --- | --- |
| `capabilities.gguf_output` | dispatch host entry | `true` allows GGUF output from a GGUF source. |
| `capabilities.gguf_hf_output` | dispatch host entry | `true` also allows an HF source. It has an effect only if `gguf_output` is `true`. |
| `capabilities.gguf_safetensors_output` | dispatch host entry | `true` allows standalone SafeTensors with registered `assets`. It is false by default and requires `gguf_output`. Production workers do not have this capability enabled. |
| `capabilities.gguf_rocmfpx_output` | dispatch host entry | `true` is required for a ROCmFPX preset or tensor override. The worker also needs `gguf_output` and the capability for the source type. |
| `BREWERY_GGUF_QUANTIZER` | engine | Absolute path of a regular, CPU-only `llama-quantize`. The engine runs it offline with a fixed, cleaned environment. |
| `BREWERY_GGUF_ROCMFPX_QUANTIZER` | engine | Absolute path of a separate, self-contained CPU ROCmFPX quantizer executable. Do not use the stock quantizer path. |
| `BREWERY_GGUF_ROCMFPX_SHA256` | engine | Lowercase SHA-256 digest of the ROCmFPX quantizer. The engine checks the digest and runs a private copy of the verified binary. |
| `BREWERY_GGUF_HF_EXPORTER` | engine | Required for HF and standalone SafeTensors sources. Unused for GGUF sources. |

ROCmFPX jobs use the independent `llama-quantize-rocmfpx` encoder. A worker
must report `capabilities.gguf_rocmfpx_output: true` and have both ROCmFPX
engine settings above. The controller binds each fork selection to the pinned
revision. The worker checks the revision and settings. The engine rejects a
binary whose SHA-256 does not match. Stock `llama-quantize` remains separate.

**Staging a local worker.** `deploy/lan/stage_local_gguf.py` requires
`--root`, `--release`, `--quantizer`, `--catalog` and `--dispatch`. Use a new
staging directory for `--root`. Pass the approved stock CPU binary to
`--quantizer`; the stager checks its pin.

For standalone SafeTensors, pass `--runtime`, `--llama`, `--probe-source` and
`--probe-assets`. Use a dedicated CPU Python runtime with PyTorch `2.11.0+cpu`
and Transformers `4.57.6`, the pinned llama.cpp checkout, and a matching
fixture (at most 32 MiB each for weights and assets). The staged worker and
engine must convert and verify the fixture before the stager proposes
`gguf_safetensors_output`. The probe uses private copies and scratch worker
state, and requires the normal RAM and disk admission limits. The worker
checks the runtime versions again for standalone jobs. Stock GGUF and weight
jobs do not require the standalone runtime.

For ROCmFPX, pass all three fork flags:

| Flag | Value |
| --- | --- |
| `--rocmfpx-quantizer` | Separate CPU quantizer from the pinned ROCmFPX source. |
| `--rocmfpx-sha256` | `593e59cb7bcdd51ca4f9e7cac3ea42797c48dd1c18ac9b13552dd8c382ae3708` |
| `--rocmfpx-probe-source` | Unquantized GGUF fixture, at most 8 MiB. |

The stager checks the supplied binary against the pinned SHA-256. It runs a
bounded, network-isolated convert and verify probe before it enables
`capabilities.gguf_rocmfpx_output`. It writes `dispatch.proposed.json` and
leaves the supplied dispatch file unchanged.

Replace the sample paths below. Use a new `/arboretum` directory for `ROOT`.
The `--catalog` value must point to this checkout's pinned catalog.

```sh
ROOT=/arboretum/rocmfpx-stage-unique-name
RELEASE=/path/to/verified-release
STOCK_QUANTIZER=/path/to/llama-quantize
ROCMFPX_QUANTIZER=/path/to/llama-quantize-rocmfpx
DISPATCH=/path/to/dispatch.json
PROBE=/path/to/tiny-unquantized.gguf
python3 -B deploy/lan/stage_local_gguf.py \
  --root "$ROOT" \
  --release "$RELEASE" \
  --quantizer "$STOCK_QUANTIZER" \
  --catalog crates/controller/src/gguf_catalog.json \
  --dispatch "$DISPATCH" \
  --rocmfpx-quantizer "$ROCMFPX_QUANTIZER" \
  --rocmfpx-sha256 593e59cb7bcdd51ca4f9e7cac3ea42797c48dd1c18ac9b13552dd8c382ae3708 \
  --rocmfpx-probe-source "$PROBE"
```

A worker that pins a source for release qualification refuses GGUF output.
The API has no route to change these settings.

**Reader and smoke check.** A GGUF file with ROCmFPX tensors needs a reader
built from the same pinned fork revision. Do not use a stock `llama.cpp`
reader for these files.

The smoke script needs NumPy and `gguf-py` from the pinned fork checkout. Use
CPU-only `llama-quantize-rocmfpx` and matching `llama-simple` builds. Set
`OUT` to a new directory under `/arboretum`, then run:

```sh
ROCMFPX_SRC=/absolute/path/to/ROCmFPX
ROCMFPX_QUANTIZER=/absolute/path/to/llama-quantize-rocmfpx
ROCMFPX_READER=/absolute/path/to/llama-simple
OUT=/arboretum/rocmfpx-smoke-unique-name
PYTHONPATH="$ROCMFPX_SRC/gguf-py" python3 -B qualification/rocmfpx_smoke.py \
  --quantizer "$ROCMFPX_QUANTIZER" \
  --reader "$ROCMFPX_READER" \
  --out "$OUT"
```

The script checks five base encodings, an exact tensor override, and CPU
decode on a synthetic Qwen2 fixture. It checks the output file structure and
block sizes. This is a format and runtime smoke check. It does not measure
model quality or GPU performance, and it is not release qualification.

**What the result proves.** The `convert` stage writes `converted.gguf`
only after a structure check. `verify` then reads the registered output on
its own and compares it with the source and the plan. The result records the
real SHA-256 and size of each input and of the output, the SHA-256 of the
quantizer program, and, in `report.gguf_tensors`, the type of each tensor that
it read from the output file. `verify` passes only if the header, the tensor
inventory, the shapes, the payload bounds and the planned types all match.
The controller registers `artifact_model_id` (a `gguf` model) only after
`verify` passes.

**What it does not prove.** `verify` is a structural check with file hashes.
It does not decode the values, load the file in a runtime, or measure quality
or accuracy. A GGUF conversion is not a release qualification.

**Fail-closed cases.** The job fails, with no output published, when:

- a preset or type needs an imatrix and none is registered;
- the quantizer logs a fallback, or a missing importance matrix;
- the output types differ from the plan, or the preset or inventory is wrong;
- the source has a quantized weight, is split, or has a pinned source digest;
- the GGUF is malformed or truncated;
- the quantizer fails, runs too long, or writes too much log;
- the disk has too little space for the output.

**Hugging Face limits.** The exporter accepts local HF snapshots and standalone
SafeTensors for two declared architectures: Qwen2 (`Qwen2ForCausalLM`) and
GPT-2 (`GPT2LMHeadModel`). Both support only dense inputs up to 512 MiB, with
exact tensor names and shapes and BF16, F16 or F32 tensors. Qwen2 must be
untied, use `model_type: qwen2`, and include a local Qwen2 tokenizer. GPT-2
must use `model_type: gpt2`, declare `GPT2LMHeadModel`, and set
`tie_word_embeddings` to true or false. It requires a local GPT-2 tokenizer
that passes the pinned exporter checks. An untied GPT-2 input also needs
`lm_head.weight`. HF input must be
a flat indexed snapshot or one `model.safetensors` file. Standalone input must
be one `.safetensors` file plus the registered `assets` directory described
above. Extra, quantized or sidecar tensors are refused. The exporter emits one
BF16 `model.gguf` intermediate without splitting it. A `convert` workflow
passes that intermediate to the selected `llama-quantize` preset. The exporter
needs a CPU PyTorch runtime inside a network-isolated sandbox. For HF and
standalone sources, `inspect` and `plan` read metadata and do not promise
output types. `verify` repeats the export in a transient file.

A scratch smoke check exported one standalone SafeTensors file with 15
synthetic Qwen2 tensors and a separately registered metadata directory. The
export produced a 25,396,320-byte BF16 GGUF. The pinned CPU `llama.cpp` reader
loaded and decoded that file.

A separate synthetic, declared, untied `GPT2LMHeadModel` fixture completed all
four workflow stages. Its 17-tensor GGUF kept the untied output head. The
pinned CPU reader decoded 50,257 finite logits. The pinned public
`tiny-random-gpt2` snapshot failed inspection because it does not declare
`GPT2LMHeadModel`. Its download receipt matches the failed input by SHA-256,
revision and size. The check produced no GGUF artifact and was not a
production run.

These checks cover only the tested inputs and BF16 output. They do not
establish universal runtime support, universal input acceptance, general
quantization coverage, model quality or release qualification.

The `qwen4_exp` Flash-Next checkpoint remains unsupported for GGUF export.
Inspection can report its text, projector, MTP and engram tensors, but it does
not write GGUF files for them. Native text-plus-`mmproj` export and runtime
pairing remain pending.

### Job kinds

`transfer`, `transfer_preview`, `cache_cleanup`, `compare`, `inspect`,
`plan`, `convert`, `verify`, `generate`, `evaluate`, `baseline`, `benchmark`.

### Model kinds

`gguf`, `hf` (alias `bf16-checkpoint`), `hgn`, `brewery-bundle`, `hgn-v2`,
`ht43`, `gsm8k-baseline`, `gsm8k-evaluation`, `gsm8k-predictions`, `assets`, `imatrix`, `ngram`,
`vision`, `mtp`.

### Job rules

The `inspect`, `plan`, `convert` and `verify` rows describe single-stage
diagnostic jobs. For a normal conversion, use
[Convert a model](#convert-a-model). The workflow takes the same inputs and
options as `convert`, and `decode_backend` for `verify`.

| Kind | Inputs and options |
| --- | --- |
| `compare` | Two `gguf` inputs, in order, and a prompt. Options: `n` (0 to 128, default 16), `raw`, `think`, `threads` (1 to 64), `labels` (0 to 600), `gpu_device`, `gpu_layers`. The default is CPU. |
| `inspect` | One checkpoint, and optional component inputs. Options: `profile`, `ht_backend`, `i4r_backend`, `q4cp_backend`. |
| `plan`, `convert` | As `inspect`, plus `seed`, `threads`, `ht_beam`. Extra inputs by index: `assets`, `ngram`, `vision`, `mtp`; see [Conversion components](#conversion-components). `convert` always writes a new `converted` directory. |
| `verify` | The checkpoint and the converted `hgn` bundle (second input), then optional component inputs, selected from index 2. Option: `decode_backend`. |
| `generate` | One HGN bundle directory with `manifest.json`, a prompt of at most 16384 bytes, and optional `max_tokens` (1 to 512, default 128). |
| `evaluate` | One HGN bundle directory. It scores the candidate by default. Optionally add one `gsm8k-baseline`, `gsm8k-evaluation` or `gsm8k-predictions` reference file, or set `options.baseline_source`, but not both. You cannot change the prompts or token limit. |
| `baseline` | One `gsm8k-baseline`, `gsm8k-evaluation` or `gsm8k-predictions` file, or `options.baseline_source`. |
| `transfer`, `transfer_preview` | A destination `host` and one or more inputs. |
| `cache_cleanup` | A target `host`, no inputs. Option: `bundle_sha256`. |
| `benchmark` | A target `host`, no inputs. Options: `benchmark_target` and `benchmarks` (1 to 8 distinct IDs). See [Managed benchmarks](#managed-benchmarks). |

## Sample JSON

### Dispatch file (controller)

One local host and one SSH host:

```json
{
  "hosts": {
    "workstation": {
      "local": true,
      "worker": "/home/me/model-brewery/target/release/brewery-worker",
      "config": "/home/me/.config/model-brewery/worker.json"
    },
    "frame": { "ssh": "frame" }
  }
}
```

Each host can also set a display `name`, `capabilities`, and `reasons`.
The web page shows a reason when an action is not available.

To distribute **independent whole conversions**, put one dispatch entry per
physical worker in the pool. For example:

```json
{
  "hosts": {
    "amd": {"ssh":"frame","quantization_pool":true,
            "quantization_backends":{"ht":["hip"],"i4r":["hip"],"q4cp":["hip"]}},
    "nvidia": {"ssh":"gpu-workhorse","quantization_pool":true,
               "quantization_backends":{"ht":["cuda"],"i4r":["cuda"],"q4cp":["cuda"]}}
  }
}
```

Submit a `convert` job with `host:"auto"` to use an idle eligible worker.
Set the encoder backends explicitly for GPU processing; routing does not turn
a CPU request into a GPU request. To qualify a particular card, pin its host.
Keep a host alias attached to the same worker and private root while a job is
active. Do not put two aliases for the same GPU in the pool.

To divide **one checkpoint/profile** among workers instead, use the
[distributed conversion command](qualification/distributed_conversion.md).
This is an operator CLI, not the **Any eligible worker** option in the web
page. It partitions and packs on the coordinator, distributes independent
encoding units, then merges and verifies the output on the coordinator.
Workers must support the selected encoder settings. Reserve capacity for the
largest unit, the retained packages, and the final merge. Use a shared GPU
admission gate if either workflow can use the same physical GPU.

### Worker file

```json
{
  "root": "/var/lib/model-brewery/worker",
  "source_roots": ["/home/me/models", "/data/checkpoints"],
  "engine": "/home/me/model-brewery/target/release/brewery-engine",
  "admission": "/usr/local/bin/gpu-thermal-run"
}
```

### Dispatch protocol

`brewery-dispatch ACTION HOST JOB_ID` where `ACTION` is `start`, `status`,
`cancel`, `preview_transfer`, `transfer`, or `cleanup`.

`start`, `preview_transfer`, and `transfer` read this JSON on stdin:

```json
{
  "kind": "compare",
  "inputs": [
    { "host": "frame", "path": "/data/models/a.gguf", "kind": "gguf" },
    { "host": "frame", "path": "/data/models/b.gguf", "kind": "gguf" }
  ],
  "options": { "n": 16, "threads": 8 },
  "prompt": "Name a prime number."
}
```

`status`, `cancel`, and `cleanup` read no stdin. Each action prints one JSON
object on stdout:

```json
{ "state": "running", "progress": "decoding token 4 of 16" }
```

`state` is one of `queued`, `running`, `succeeded`, `failed`, `cancelled`, or
`missing`. A `succeeded` reply has a `result` field. A `failed` reply has an
`error` field. If you send `start` again for the same job ID, the worker
reports the status of that job. It does not start a duplicate.

## Compare quantized builds

Use **Compare → Build analysis reports** to open either report locally.
Start from saved controller job details, not display names or filenames:

```sh
python3 qualification/build_diff.py reference-convert.json candidate-convert.json > builds.json
python3 qualification/paired_quality.py reference-eval.json candidate-eval.json \
  --left-predictions reference-predictions.json \
  --right-predictions candidate-predictions.json > quality.json
```

The build diff compares recorded file digests, including sidecars and
assets. Equal file sizes do not mean equal weights. The quality report
checks both prediction files and scores the same 1,319 GSM8K questions.
Read the gains and regressions, not just the net score. The exact McNemar
test uses questions where correctness changed; a large p-value does not
establish equal quality. Check source, tokenizer, loaded artifacts,
runtime, and protocol before attributing a change to quantization.
The viewer keeps both reports separate; their order is not proof of a match.

## Agent interface

Agents can operate the workbench without a browser. Use
[`brewery-weight-space`](.opencode/skills/brewery-weight-space/SKILL.md) for
traces, Fisher sensitivity, scratch edits, checks, and publish requests. See the
[project skills](.opencode/skills/) for other workflows. The running
controller serves its [OpenAPI contract](api/openapi.json) at
`GET /api/openapi.json`. Use the same loopback address and worker permissions
as the web UI.

### MCP for weight-space work

Use an MCP client that supports Streamable HTTP. Set this configuration in
the client (example for Claude Code):

```json
{
  "mcpServers": {
    "model-brewery": {
      "type": "http",
      "url": "http://127.0.0.1:4141/mcp"
    }
  }
}
```

For LAN or tailnet access, use the workbench HTTPS URL with `/mcp`. The
client must trust the gateway certificate. See the access risk under
[Web settings](#web-settings).

MCP uses the workbench trust boundary and has no separate token. The
controller checks the exact Host, Origin (if sent),
and `Sec-Fetch-Site` (if sent) on the same loopback listener. The optional
gateway also checks source CIDR, Host, and Origin. Every client in an
allowed LAN or tailnet CIDR can reach `/mcp`. It can create,
edit, or discard scratch revisions, run traces and checks, and request
publication. MCP cannot approve a publication, change settings, administer
workers, or deploy.

1. Use `tools/list` to get the input schemas. Use `register_model` to add a
   worker-side file to the Library. Use `list_models` and `list_scratch` to
   find inputs and scratch revisions.
2. Use `start_trace` for a Library GGUF or a scratch revision with private
   GGUF. Put its ID in `input_ids`. Use `get_job` to poll progress
   (`wait_ms` can wait up to 5000 ms per call). Use `read_trace` with
   `manifest`, `tensor`, `logit_lens`, or `model_logits`. Use
   `create_intervention` and `new_intervention_version` to save rules.
   Use `list_interventions` and `get_intervention` to read saved rules.
   Use `create_direction_intervention` to save projection or subtraction
   rules from a verified direction artifact.
   `export_control_vector` returns a relative URL for the same workbench.
3. Use `create_scratch` and `apply_edit` for a SafeTensors edit. Use
   `merge_models` to merge pinned revisions in a new scratch child. Use
   `convert_scratch_gguf` on the parent and child. Run `run_checks` on the
   child and use `get_job` to see the result. Use `diff_tensors`,
   `diff_traces`, and `diff_behavior` to compare results. Use
   `discard_scratch` to remove a scratch revision that you do not need.
4. Use `request_publish` with the exact revision hash and check bundle
   acknowledgement if it is necessary. The operator reviews the request
   separately. Results are not release-qualified.

For every write tool, set `idempotency_key` before the call. Use a new key
for each intended write. Use the same key and arguments for a retry.
The controller uses its normal admission, validation, and replay rules.

Tools that submit to `/api/jobs` or `/api/conversions`, including
`start_trace` and `steer_feature`, accept the optional boolean
`allow_duplicate`. The default is `false`. If the controller finds
matching work, the tool returns the existing `id` and `duplicate_of: true`.
No new job starts. To submit the same work again on purpose, set
`allow_duplicate` to `true` and use a new `idempotency_key`. The tool sends
`Brewery-Allow-Duplicate: 1`. The controller still applies the admission
and same-key replay rules.

Use `prune_model` with a scratch ID, its exact hash, and a `pruning` plan.
The result is a new scratch child with a reversible delta. Rates use basis
points: 1000 means 10%. The rate applies to each selected tensor or set of
groups. It does not specify model-wide sparsity. The total size of the
edited tensors cannot exceed 32 MiB.

- `unstructured` ranks each listed matrix by absolute weight.
- `mlp_neurons` ranks Qwen2 down-projection input columns by the sum of
  absolute weights across each column. The mask zeros entire columns.
- `attention_heads` ranks groups of columns in the Qwen2 attention output
  projection by the sum of absolute weights. Set `heads` to
  `num_attention_heads` from the source configuration. The mask zeros each
  selected query head's contribution without changing shared key/value
  heads.
- `wanda_mlp` ranks Qwen2 neurons with a Wanda-style score. First convert
  the parent to private GGUF. Run a trace of that exact parent with no
  interventions. Select 1 to 64 consecutive `ffn_inp` positions from one
  prompt at the requested layer. Set `rms_eps` to `rms_norm_eps` from the
  source configuration. The editor applies RMS normalization to those
  rows before using the activation norms to scale the absolute gate and
  up weights. It zeros the selected down-projection columns.

Dense masks do not reduce the model dimensions, file size, or compute cost
in a dense runtime. Use the same private conversion, checks, and diffs as
for a direct edit.

### Native directional ablation

Use `ablate_direction` or `POST /api/scratch/{id}/ablate_direction`. Use
the scratch parent's exact `sha256`. In `direction`, give the `id`,
`version` (2), and `layer` from `get_direction`. Set inclusive
`first_layer` and `last_layer` (at most 64 layers). Select a `variant` and
a `kernel` with `shape`, `peak`, `width`, and `min_weight`:

- `flat` uses weight 1 at all selected layers. `gaussian` uses
  `exp(-0.5 * t^2)` and `triangle` uses `max(0, 1 - t)`, where `t` is
  the distance from `peak` divided by `width` in layers. `min_weight`
  is the floor (0 to 1). A zero weight skips the layer.
- `plain` subtracts the component along the fitted direction.
- `projected` first removes the direction's component along the source
  layer's training negative mean. The edit leaves the component along
  that mean unchanged.
- `norm_preserving` also projects the direction away from the target
  layer's training negative mean. It restores each writing vector's
  original L2 norm after subtraction. This uses rows of the transposed
  SafeTensors matrix (columns of the stored `[out, in]` matrix), so
  restoring the norms does not reintroduce the removed direction.

Projected variants need `negative_mean` in the artifact. Fit the
direction again if an older artifact does not have this field. For
`norm_preserving`, include all target layers in the fit. Optional `alpha`
multiplies the kernel weight. The default is 1; the allowed range is
greater than 0 and at most 4. An effective weight of 1 removes the
component along the direction, up to storage rounding. A lower weight
removes only part of it. A higher weight reverses the component.

The editor changes `self_attn.o_proj.weight` and `mlp.down_proj.weight`
for each active layer of a dense HF checkpoint. It rejects biased
writers, incompatible widths, tensors over 32 MiB, and more than
256 MiB of input tensors in one edit. Embeddings are not edited
(`include_embeddings: true` is rejected); Qwen2.5-0.5B embeddings exceed
the per-tensor limit.

Use a direction measured on the same model in the same residual basis.
The editor checks tensor compatibility, not equivalence between the
Library GGUF and scratch SafeTensors. Provenance records both pins, the
source layer, and the edit settings. The child has a reversible delta.
Convert it to private GGUF, run checks, and review the diffs before a
publish request. This workflow does not qualify a release.

References: [Arditi et al. (2024), section 4.1](https://arxiv.org/abs/2406.11717);
[Labonne's Gaussian layer kernel](https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-v2);
[Lai (2025), Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration)
and [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).

### Native HGN v2 scratch edits

Use `ablate_direction` with the ID and SHA-256 of a standalone HGN v2 Library
model on the approved source worker. Use `plain`, a `flat` kernel, and alpha 1.
Fit a version 2 direction on the Library model. The job edits attention output,
FFN down projections, and selected PLE value projections. It edits dense and
routed writers together. It does not edit embeddings or MTP weights.

Poll `get_job` and pass `result.scratch_id` to `get_scratch`. For a later edit,
use the HGN scratch ID and its exact SHA-256. Continue to use a direction fit on
its Library root. The child has no Library record or reversible delta.
`apply_edit`, conversion to GGUF, and publication do not support HGN scratch.
Discard removes access to the revision. Private worker files and cache entries
remain on Frame; remove them only when they are no longer required for audit,
resume, or an active job.

The worker stores `hgn-feedback.json` with the child. `get_scratch` exposes the
same report under `provenance.ablation.hgn.report`. It records the model hashes,
direction, encoder hash, store, variant, payload hash and size, number of
passes, the first and final decoded component fractions, and relative error.
The tolerances are 0.05 for I4r, Q4Cp and Q8G64, and 0.08 for HT3 and HT4.
BF16 uses one projection before rounding and measures any rounding residue.
The job fails if any selected tensor misses its tolerance. Completed payloads
are reused when the parent, layers, encoder, and direction match. Edit-aware
rounding is not supported. This receipt does not replace an independent
decoded-store audit or a CPU forward check.

Run both stored-file checks with `run_checks` or `POST /api/scratch/{id}/checks`.
Send the child SHA-256 without `behavior_direction`. Poll the job, then read
`get_checks` or the Edits view. In `hgn_stored`, the audit measures each
dense and routed writer from the parent and child files. A tensor with a mean
absolute parent component below `1e-6` has status `no_component` and cannot
show a drop. Each edited group (dense and routed experts) must have at least
one passing tensor. At most 10% of the measurable component can remain in
each writer and routed expert. For all writers, the relative Frobenius error
against the parent must be at most `0.10` and the mean absolute component
along each of five random directions (seed 26842) can change by at most 5%.
Missing writers, hash mismatches or disagreement with the stored feedback
fail the check. The CPU runtime uses the root model's
pinned assets for both files. It reads the last prompt position for each of
two fixed prompts and reports full-vocabulary KL(parent || child), top-1
agreement and logit deltas. It does not generate text. Scores are not behavior
gates and are not release-qualified.

On the non-Hy4 Frame HGN worker, use the HGN trace worker's approved source
roots and a private worker state directory. Set the dispatch capability
`scratch_hgn_ablation` only for this worker. Enable `trace` with the approved
HGN CPU runtime to run stored-file checks. Add an `hgn_edit` object to the
worker configuration with `host`, `grant_key_file`, and `build_admission`.
Put a shared random 32-byte key in private, regular files owned by the controller
and worker identities. Set `BREWERY_HGN_EDIT_KEY_FILE` on the controller to
its copy. Keep both files out of all model roots. Do not put the key in a job
request. The worker state directory must have mode 0700; leave the ordinary
`admission` and Hy4 settings unset.

Use an absolute path to an operator-owned build wrapper for `build_admission`.
The worker calls it as `wrapper -- /path/to/brewery-engine`. The wrapper must
hold the exclusive Frame build reservation from before source hashing until
the engine exits. It must send queue messages to stderr and keep stdin and
stdout for the engine. Do not nest a reservation. Coordinate the build window
before you enable the capability or enqueue an edit. Keep the stock model and
production units unchanged.

### Native ablation search

For a Library HGN v2 checkpoint, use `search_ablation` or
`POST /api/scratch/{id}/search_ablation` with the Library ID. Supply
`sha256`, `direction_id`, `source_layer`, `seed`, and one to four unique
`targets` from `residual`, `dense`, `experts`, and `all`. Supply both layer
endpoints for an inclusive range, or omit them to test every layer. Use a
non-Hy4 source worker with `trace`, `behavior_direction`, the pinned HGN CPU
runtime, and the contrast cache. The screen has a four-hour runtime limit.

The HGN screen compares the positive and negative sides of the reviewed
held-out pairs. It reports the stock marker, matched random-direction
scores, control-set shifts, KL, and the Pareto front. For `response_length`,
the marker is first-token non-EOS probability. It is a continuation proxy,
not a measure of generated length. The HGN screen generates no text and
creates no scratch children. Read the receipt with `get_job`.

#### GGUF search

Use `search_ablation` or `POST /api/scratch/{id}/search_ablation` on an
unrequested dense Qwen2.5 scratch revision with a private GGUF. Fit a
version 2 `l_out` direction at the last user token on a Library GGUF
with the exact same SHA-256.
Fit at least three nonzero layers. Include intermediate target layers
for the norm-preserving variant. Only slop and sycophancy are supported.

Supply `sha256`, `direction_id`, `assets_id`, a 32-bit `seed`,
`trial_budget` (3 to 32), `max_children` (1 to 3, at most the trial
budget), `disk_budget_bytes`, and a `behavior_direction` check config.
The target behavior's pinned confirmation cache must have 32 unique
held-out groups, with no overlap with A1 fit or screen prompts. The
other prompt-set pins are the usual A2 inputs. The disk check charges a
conservative allowance for each retained child, its GGUF and delta, and
one conversion's temporary files. It also checks available disk space.

Configure `BREWERY_LANTERNFISH_ABLATION` for the local CPU editor worker
with the sealed Lanternfish `1bcb466` executable, SHA-256
`d2c38124fad896149456869b02adccc78827ba9dcc9657f88efa6e869d8d2b92`.
Set `BREWERY_LANTERNFISH_PROJECTION_LIB_DIR` to the shared A4 library
directory. Configure the contrast cache, BF16
converter and W4 checker as usual. No GPU trace backend is used.

The screen edits activations, not weights. It scores the original A1
holdout at the last user token, the last four user tokens, or the last
user and assistant-prefix tokens. For sycophancy it reads the first-token
logit difference between the agreeing and disagreeing answers, using the
Qwen `(A` and `(B` tokens. For slop it chooses `Certainly`, `In`, or `It`
by the largest difference in first-token probability between the stock
model's positive and negative A1 training prompts. For admission, the
slop gap must be positive in training and the opener probability at
least 5% on one screen prompt. Sycophancy requires at least one stock
prompt with a positive agreement margin. These are screening markers,
not rates of generated behavior.

The seeded search tries different source layers, target ranges, Gaussian
widths, variants, scales, and position sets. It uses the marker reduction
and the full first-token distribution KL from the parent. Each trial has
a same-norm seeded random direction with the same settings. The result
keeps the Pareto front. A front point can make a scratch child only when
it lowers the marker and beats its matched control on mean reduction and
on a majority of prompts. No other point writes weights. The screen
applies projection to the residual; the weight edit applies it to the
residual writers. Their effects can differ.

Read the queued job with `get_job` or `GET /api/jobs/{id}`. Check the
marker, random control, KL, child W4 checks and A2 rates. The generated
behavior comparisons have Wilson 95% intervals and exact paired McNemar
tests, with Holm correction across the selected children in that search.
No result is release-qualified. The job does not request publish or
approve a child.

Use `plan_experts` with a registered local GGUF and two independent
`brewery.expert-activations.v1` JSON observations to generate a REAP map.
The planner needs Python 3 at `/usr/bin/python3` on the controller. It
checks the GGUF hash and does not run inference or write model weights.
The plan is not proof of quality. Keep the exact map text and SHA-256.
`prune_experts` accepts a reviewed map but returns 503 until a verified
private scratch writer is available. It does not fall back to an operation
that adds an unapproved revision to the Library.

The server uses MCP protocol `2025-06-18`. It accepts one JSON-RPC message
per POST. It does not keep sessions, open an MCP event stream, or offer a
stdio transport. POST bodies are limited to 256 KiB. Tool data is limited
to 4 MiB from the controller and 8 MiB for the whole MCP response. Use a
smaller trace slice or read the [controller API](api/openapi.json) directly
if a result is too large.

### Hosted workbench agent

1. In Settings, select Anthropic or an OpenAI-compatible provider. Set the
   model ID and API base URL. For example, use `https://api.anthropic.com/v1`
   or `https://api.openai.com/v1`. Do not include `/messages` or
   `/chat/completions`. Use HTTPS. The controller accepts HTTP only on a
   numeric loopback address for local providers and tests.
2. Save the settings, then set the API key. Settings shows only whether a
   key is set. Use the same control to replace or remove it. A change to
   the provider kind or base URL removes the key. The controller does not
   test the key when you save it, use a proxy, or follow redirects.
3. Open Agent, select the Library model, and enter the goal. The controller
   sends the goal, model ID, MCP tool schemas, and conversation to the
   selected provider. Tool results can include model paths, prompts, and
   trace or check data. Use a provider that may receive this data. Do not
   put secrets in goals, prompts, or model records.
4. Follow the transcript and open scratch revisions in Edits. The agent
   uses the same tools and admission rules as `/mcp`. It can request
   publication, but only an operator can approve it in Edits. Results are
   not release-qualified.

Only one session can run at a time. Settings limits the number of hosted
requests, tool calls, and output tokens per request. The session token
limit uses the input and output counts that the provider reports. A
request can exceed the remaining budget because input token use is known
only after the reply. If this occurs, the controller does not execute
tools from that reply. The request timeout and the session time limit
also stop the loop. Cancel stops new requests and tool calls. A tool
already admitted can finish, and queued jobs continue. Cancel those jobs
separately when necessary.

The controller keeps the session transcript and tool results in its
database. Agent shows the most recent 100 sessions. A restart does not
resume an interrupted session. The provider key is in a separate
owner-only file, not the database. Do not serve the controller data
directory or include private credential files in exports. Provider
settings and sessions use the same trust boundary as the workbench.
Every client with workbench access can use the configured provider and
read session transcripts.

### JSON command-line client

```sh
tools/brewery-agent describe
tools/brewery-agent state
tools/brewery-agent paths frame /data/models/
```

For writes, choose a key **before** making the request and save it with
the request JSON. Use a new key for each intended creation. For example:

```sh
tools/brewery-agent create jobs --key experiment-42.preview --file preview-job.json
tools/brewery-agent wait "$JOB_ID" --timeout 300
```

To convert a model, create one conversion workflow. Then poll it, or cancel it:

```sh
tools/brewery-agent create conversions --key convert-a.v1 --file conversion.json
tools/brewery-agent conversions
tools/brewery-agent conversion "$CONVERSION_ID"
tools/brewery-agent cancel-conversion "$CONVERSION_ID"
```

The workflow runs `inspect`, `plan`, `convert` and `verify`. Follow the child
jobs with `progress` and `job`, using the IDs in `job_ids`. The workflow adds
no event kind. Use the job events.

`tools/brewery-agent progress "$JOB_ID"` shows the current phase without
waiting for completion. The Jobs page shows the same counts and the current
file when available. A percentage describes **one phase**, such as hashing
the input, not the entire conversion. A phase with no known total has no
percentage or ETA. Progress displays must pair the phase with its units:
show bytes processed and decimal MB/s for a byte pass, or the reported
count and corresponding rate for counted work. A rate needs comparable,
timestamped samples from the same phase; do not carry it into a new phase
or invent one for an unmetered operation. When the total is known and
recent samples show a positive rate, the web UI estimates the time left
in that phase: remaining units divided by units per second. This is not
the time left in the entire job. The live page labels read-only
process observations separately from worker reports and drops them when stale.
On the Jobs page or a running job's detail page, expand **CPU thread
activity** to see the current subtask for each reported worker slot. These
are assignments from the worker pool, not physical core numbers. A phase
that does not report assignments has no list; the UI does not infer them.

For cross-host input progress, install matching dispatcher and worker
builds on both ends after active jobs finish. Older workers do not
implement the streaming request. Native conversion counters also require
a converter built with `BREWERY_NATIVE_PROGRESS` support.

If a Frame Hy4 job reports only `Converting model (…s elapsed)`, the
worker has not reported a work counter. Elapsed time is not proof that
another byte was converted. Do not estimate progress from the size of
the partial GGUF: the encoder sets its final length before it copies
the payload, then reads the data back to verify it. A full-size partial
file does not show that either pass is complete. Leave the job running;
updating the UI cannot recover counters that its worker never sent.

If the POST response is lost, repeat the same request with the same key.
A replay returns the original ID, not a second job. A `409` means the
key was already used for a different request. `wait` prints the terminal
job detail; exit 4 means failed or cancelled, and exit 5 means the client
timed out without cancelling the job.

For multiple jobs or clients, read `event_cursor` from `state`, then run
`tools/brewery-agent events --after "$CURSOR"` (or use `GET /api/events`
with `Last-Event-ID`). Save the last **processed** event ID. Events say
what changed; fetch the job detail to read progress or results. Resume
from the saved ID after a disconnect. On `resync` (CLI exit 3), take
a fresh state snapshot and reopen from its cursor instead. The bounded
stream is not an archive or a substitute for job receipts.

Use `tools/export-scorecard` on saved benchmark/evaluation job details.
Calibration reports, distributed partition/merge, and release evidence
collection have local CLI entry points listed in the skills. Worker
configuration, credentials, file access, and GPU admission stay under
operator control; the API does not expose an arbitrary command runner.

### Controller HTTP

All JSON POST requests need `Content-Type: application/json`.

`GET /api/state` reports model availability from the owning worker. These
checks run in the background. Registrations for the same worker, executor,
and path share one result for 30 seconds. A missing or disallowed path, or a
disabled or removed worker, reports `available: false`. If the ordinary
worker cannot check access on a separate weight editor, the controller
reports `available: null`. Pending, failed, and unsupported checks also
report `null`. Update older dispatchers and workers to get file state.
The check does not hash or read model data. A reported `sha256` comes from
a verified output receipt, not from the file check.

`GET /api/jobs/{id}` returns `prompt_text`, the text sent to the worker at
submission. Later edits to a saved prompt do not change this text.

```sh
curl -s http://127.0.0.1:4141/api/state
curl -s -X POST http://127.0.0.1:4141/api/suggestions/paths \
  -H 'Content-Type: application/json' \
  -d '{"host":"frame","prefix":"/data/models/gran"}'
curl -s -X POST http://127.0.0.1:4141/api/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"model-a","host":"frame","path":"/data/models/a.gguf","kind":"gguf"}'
curl -s -X POST http://127.0.0.1:4141/api/prompts \
  -H 'Content-Type: application/json' \
  -d '{"name":"prime","text":"Name a prime number."}'
curl -s -X POST http://127.0.0.1:4141/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"compare","host":"frame","input_ids":["<model-id-1>","<model-id-2>"],"prompt_id":"<prompt-id>","options":{"n":16}}'
```

To convert a model, send one request to `/api/conversions`. Use the IDs of a
registered source and of any auxiliary inputs:

```sh
curl -s -X POST http://127.0.0.1:4141/api/conversions \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: convert-a.v1' \
  -d '{"host":"auto","input_ids":["<source-model-id>"],"options":{"profile":"v2"},"decode_backend":"cuda"}'
```

A new workflow returns `201`. The same key and the same body return `200`
with the current workflow. A different body for the same key returns `409`.

For `/api/jobs` and `/api/conversions`, the controller checks the
`Idempotency-Key` first. Then it compares a SHA-256 fingerprint of the
canonical request body on the same route. It does this with or without a
key. If matching work is queued, running, or cancelling, or if the
controller accepted it less than 10 seconds ago, the controller does not
queue new work. It returns `200`, the existing ID, `duplicate_of: true`,
and the `Brewery-Duplicate` header with that ID. A new key on a match is
bound to the existing ID.

SQLite stores the fingerprint in the same transaction that queues the
work. This check still works after a controller restart. Work queued
before this feature was installed has no fingerprint. The controller
removes expired fingerprints for finished work on later submissions.
To repeat identical work on purpose, select **Queue another** after a
match, or send `Brewery-Allow-Duplicate: 1` with a new `Idempotency-Key`.
The normal admission checks still apply.

If a reply is lost, check Jobs or retry with the same inputs. The
workbench keeps an unconfirmed key during the current visit. It treats
a `4xx` reply as a controller rejection only if the reply has
`Brewery-Error: 1`, `Content-Type: application/json`, and exactly one
string field named `error`.

Settings and administration routes are in
[Web settings and administration](#web-settings-and-administration).

Other routes: `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`,
`GET /api/conversions`, `GET /api/conversions/{id}`,
`POST /api/conversions/{id}/cancel`, and `DELETE /api/models/{id}`. Errors
use `{"error": "..."}`.

### Spectral weight health

In Edits, select a scratch revision and open Weight health. Start with one
layer or tensor. The revision uses its direct parent as the reference.
Select a different reference to compare with another revision or Library
quantization on the same host. The job needs an enabled local SafeTensors
editor. Use `representation: "gguf"` for an existing private scratch GGUF.
The default reference remains in SafeTensors format.

With REST, send `POST /api/weight-spectra` with `id`, and optionally
`compare_id`, `representation`, `compare_representation`, `layer`, and
`tensor`. Use an exact tensor name. Use a Library ID in `id` to measure that
model without creating a scratch revision. The MCP tool is `weight_spectra`.
Use `GET /api/jobs/{id}` or `get_job` with the returned job ID. The result
contains the source and reference SHA-256 values. For sharded SafeTensors
it also lists the sorted shard names and hashes. Deltas are source minus
reference. A missing tensor or a shape mismatch has no delta.

The CPU job reads SafeTensors F32, F16 and BF16, and single-file little-endian
GGUF V2/V3 F32, F16, BF16, Q4_0, Q8_0, Q4_K and Q6_K. Other types and
tensor ranks above two do not have measurements. The job compares matching
names, with aliases for common Llama and Qwen attention, MLP and norm weights.
A report can include at most 512 tensors.

Small tensors use exact SVD. Large matrices use a seeded randomized SVD
with up to 32 singular values. Effective rank is the exponential of Shannon
entropy of normalized singular values. Stable rank is the full squared
Frobenius norm divided by the largest squared singular value. Use the
captured energy fraction to see how much of the full spectrum the
randomized result covers. Effective rank and alpha from the randomized
result use only the retained values.

The WeightWatcher-style alpha follows Martin and Mahoney, [Implicit
Self-Regularization in Deep Neural Networks](https://jmlr.org/papers/v22/20-410.html)
(JMLR, 2021, section 4). It fits a continuous power law to the squared
singular values. It uses maximum likelihood and selects the lower cutoff
with the smallest Kolmogorov-Smirnov distance. The fit needs at least eight
nonzero tail samples. The fitted exponent alone does not show that a
tensor has a power-law spectrum. Results are not release-qualified.

### Model lineage

In Library, open Model lineage and select the model to test. Select up to
three candidates on the same local host. If there are three or fewer
eligible entries, you can leave the candidates blank. To test a merge,
select the common base and both possible parent models.

Call MCP `lineage` with the target Library `id` and optional `candidate_1`,
`candidate_2`, and `candidate_3` Library IDs. The REST route is
`GET /api/models/{id}/lineage` with the same optional query arguments.
The read does not create a job, add a record, or change a checkpoint.

Use models in approved local directories or approved scratch models that
are already in Library. The check accepts F32, F16, and BF16 in standalone
or sharded SafeTensors and in floating-point GGUF. Compare the same format
and storage dtype. Matching tensor names, shapes, and types serve as the
architecture check. This does not compare tokenizers or model configs.
Each checkpoint can be up to 8 GiB. Only one read can run at a time.

Use `most_likely` for the suggested base and relation. Read `candidates`
for input hashes, changed fractions, zeroing evidence, and bounding boxes.
Bounds start at zero and exclude the end. The check uses the X2 spectrum
of the delta for up to three changed tensors with the most delta energy.
It does not calculate spectra for unchanged or oversized tensors. A
merge fit reports parent IDs, weights, agreement of parent directions,
and the relative residual of the fit. It does not prove which merge tool
produced the weights.

The confidence score measures agreement with the fingerprint, not a
calibrated probability. The base is null when no candidate is close
enough or there is no clear winner. Weight changes alone cannot prove
causal direction or distinguish every fine-tune from a manual edit.
Quantized and cross-format comparisons are not supported. Results are not
release-qualified.

### Scratch weight revisions

Enable the local CPU weight editor for the worker that owns the source.
Register a standalone `.safetensors` file in an approved model directory.
Use its Library ID and SHA-256 to create a scratch revision. You can also
use the ID and SHA-256 of a scratch revision as the parent. Each edit makes
a new child. To roll back, select the parent ID in `provenance.parent_id`.
Do not delete the Library entry if you need to select that parent later.

| Method and path | JSON request | Result |
| --- | --- | --- |
| `POST /api/scratch` | `{"parent_id":"<id>","parent_sha256":"<hash>"}` | New scratch record with an empty delta (`201`). |
| `POST /api/scratch/{id}/edits` | `{"operations":[...]}` | New scratch child of `{id}` (`201`). |
| `POST /api/scratch/{id}/rome`, `/memit`, `/alphaedit` | Parent `sha256`, `layers`, `facts`, optional `steps` and `backend` (`cpu`, `cuda`, or `rocm`). | Queued edit (`202`). Read `result.scratch_id` when it succeeds. |
| `GET /api/scratch` | None | List of scratch records with provenance (`200`). |
| `GET /api/scratch/{id}/delta` | None | Parent and child hashes, delta hash and size, and counts of changed tensors and bytes (`200`). |
| `POST /api/scratch/{id}/gguf` | `{"sha256":"<hash>","assets_id":"<Library assets ID>"}` | Private BF16 conversion job (`202`). |
| `POST /api/scratch/{id}/checks` | `{"sha256":"<hash>"}` | Worker check job (`202`), or an integrity-only result (`200`). |
| `GET /api/scratch/{id}/checks` | None | Current check bundle (`200`), or `404` if there is none. |
| `POST /api/scratch/{id}/discard` | `{}` | Record with state `discarded` (`200`). The controller removes the files and omits the record from the list. Discard unapproved children before their parent. You cannot discard an approved revision. |
| `POST /api/scratch/{id}/publish-request` | `{"sha256":"<hash>","acknowledge_bundle":"<bundle-id>"}` | Record with state `publish_requested` (`200`). The Library does not change. |
| `POST /api/admin/scratch/{id}/approve` | `{"sha256":"<hash>"}` | Record with state `approved` and `library_model_id` (`200`). |

Send an `Idempotency-Key` with each POST. If the response does not arrive,
send the same request with the same key. A matching retry returns the current
record with status `200`; a different body or target with the same key returns
`409`. The controller runs one scratch write at a time. On `503`, wait and
retry with the same key.

Send operations in order. `set` and `add` use zero-based `origin` and `shape`
for a rectangular slice. Give exactly one value per cell in row-major order.
`low_rank` adds the outer product of `u` and `v`, multiplied by `scale`, to a
matrix. `low_rank_batch` takes lists of paired `u` and `v` vectors. It adds
their outer products in F32, then writes the matrix once in F32 or BF16.
For `orthogonalize`, use `rows`
to remove the projection from each row, or `columns` to remove it from each
column. Give a nonzero `direction` with one value per element of a row or
column, respectively. The result rounds to the stored dtype after each
operation. The editor rejects non-finite results and batches with no stored
change. For example:

```json
{"operations":[
  {"op":"set","tensor":"w","origin":[0,1],"shape":[1,2],"values":[8,9]},
  {"op":"add","tensor":"w","origin":[1,0],"shape":[1,2],"values":[1,2]},
  {"op":"low_rank","tensor":"w","u":[2,1],"v":[3,0,2],"scale":0.5},
  {"op":"orthogonalize","tensor":"w","axis":"rows","direction":[1,0,0]}
]}
```

The source limit is 4 GiB. One request can edit up to 32 MiB of tensors
in up to 32 operations. Editable dtypes are F32, F16 and BF16. The controller
keeps at most 64 scratch revisions that are not discarded. It copies the full
file for each child and checks the local editor's disk reserve. Indexed HF
directories and SSH workers are not supported by this API.

#### Merge revisions

Use `POST /api/scratch/merge` or the `merge_models` MCP tool. Set
`base_id`, `base_sha256`, `algorithm`, and `inputs`. Each of the 1 to 16
ordered inputs has an `id`, `sha256`, and `weight`. Use scratch revisions
or standalone SafeTensors Library entries on the same enabled local editor.
Tensor names, shapes, and dtypes must match. The 4 GiB source limit applies.
Each file can have at most 4096 tensors. The editor uses F32 math and saves
the result in the base dtypes (BF16, F16, or F32).

- `{"method":"linear"}`: Weighted average (model soup). Use nonnegative
  weights with a positive sum. Include the base in `inputs` to give it a
  weight in the average.
- `{"method":"task_arithmetic"}`: Add the weighted input deltas to the base.
- `{"method":"ties","density":0.4}`: Keep the largest magnitudes in each
  input tensor delta before weighting. Use the first flat index at a
  magnitude tie. Elect the sign of the weighted sum (positive on a tie).
  Add the weighted mean of the deltas with the elected sign to the base.
- `{"method":"dare_linear","drop_rate":0.5,"seed":2026}`: Drop delta
  elements at random and scale the survivors by `1 / (1 - drop_rate)`.
  Add the weighted sum to the base. `dare_ties` also elects a sign and
  adds only the weighted deltas with that sign.

DARE uses SplitMix64. It gives the same output for the same base layout,
ordered inputs, weights, and seed. It does not use the PyTorch RNG or its
observed-norm rescale. Keep the input order and hashes for a replay.
The merge reserves space for the full child and a delta as large as its
payload. To run paired checks, use a scratch base with its private BF16
GGUF. The output remains scratch and **not release-qualified**.

#### Locate and edit a fact

Convert the parent to private BF16 GGUF first. Configure the two
`BREWERY_LANTERNFISH_EDIT` settings on the local worker for the default CPU
backend. The engine checks the binary and the four shared libraries against
the compiled pins before launch. The W0 trace pin is separate.
The private converter supports Qwen2 and bounded dense Llama models with
local BPE tokenizer JSON and unscaled RoPE. Larger models and other Llama
variants may exceed the conversion or tensor limits.

Use `rome_edit`, `memit_edit`, or `alphaedit_edit` in MCP. Put one `{}` slot
in each `prompt`. Enter increasing, zero-based `layers`. ROME takes one
fact and one layer. MEMIT takes up to eight facts and two or three layers.
AlphaEdit takes up to eight facts and one to three layers. The optional
`steps` parameter is 1 to 32 (default 16). The 32 MiB tensor limit also
applies to these jobs. Success makes a child without changing the parent
or the Library. Use the normal private conversion and check tools on it.

To use a GPU, set `backend` to `cuda` or `rocm` in the API or MCP request.
Configure the original local worker as follows:

- Set `admission` to the absolute path of the GPU admission wrapper. Use
  the same lock as other GPU jobs on this host.
- In `engine_env`, set `BREWERY_ALLOW_GPU_LOAD` to `"1"`.
- For CUDA, set `BREWERY_LANTERNFISH_EDIT_CUDA` to the absolute path
  of the W14b executable and `BREWERY_LANTERNFISH_EDIT_CUDA_LIB_DIR` to the
  absolute directory of its five pinned libraries. See the
  [W14b pin](qualification/lanternfish-w14b-cuda.md). You can set
  `CUDA_VISIBLE_DEVICES` to restrict the available devices.
- For ROCm, set `BREWERY_LANTERNFISH_EDIT_ROCM` to the absolute path of
  the pinned executable and `BREWERY_LANTERNFISH_EDIT_ROCM_LIB_DIR` to the
  absolute path of `lib` in that package. Keep the `rocblas/library`
  subdirectory with its Tensile files. See the
  [ROCm pin manifest](qualification/lanternfish-rocm.sha256).
  To restrict devices, set `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`.
  Omit the other mask. Allow 768 MiB of disk space for staging.

Set the runtime paths and device masks in `engine_env`. Each GPU edit uses
only its own runtime settings. The engine verifies the executable and
library hashes and stages private copies before launch. Forward and
backward compute require GPU placement. CUDA also disables TF32. If no GPU
is visible for the requested backend, the job fails. It does not fall back
to CPU. The scratch provenance gives the backend and device, for example
`rocm` and `ROCm0`.

The worker expands BF16 GGUF to F32 for ggml backward. The W3 editor writes
the update in the original SafeTensors dtype. The private forward pass
rounds each step to the private GGUF dtype before it solves for the next
layer. ROME uses a rank-one update, MEMIT updates several layers in
sequence, and AlphaEdit projects the update away from the sampled
preserved keys. The sample does not represent all facts. Read the measured
checks before you use an edit. See [Stage 2 acceptance](qualification/weight_space_stage2.md).

#### Checks before a publish request

Register a Library `assets` directory with the approved local worker. It must
contain the model configuration and tokenizer files, not a copy of the weights.
Use the same assets for both conversions. Poll each conversion with
`GET /api/jobs/{id}`. Read the GGUF SHA-256, SafeTensors SHA-256, and converter
version in the scratch record. The GGUF is not a Library file. Ordinary jobs
cannot use it. Only the assigned scratch job gets access. Discard removes the
private file after the scratch jobs complete.

Enable `trace` on that worker to trace a scratch revision. Use its ID in
`input_ids`, as for a Library model. Bind a saved intervention to that scratch ID
and its GGUF SHA-256. Patches from another trace must have the same ID and
GGUF SHA-256. To compare traces, capture the same prompts and token positions
with the same capture options.

Run checks before you send a publish request. The controller checks the
parent and child hashes, tensor names, shapes, dtypes, offsets, and finite
values against the immediate parent and edit record. Read `integrity.status`.
You cannot waive a failed integrity check.

If integrity passes and both GGUF files are ready, the worker loads both
models and generates text. It measures target, paraphrase and locality
results on fixed CounterFact and zsRE subsets. It measures perplexity on a
separate corpus. Poll the job ID, then read `GET /api/scratch/{id}/checks`.
The measured report identifies both SafeTensors and GGUF files by SHA-256.
Use `GET /api/diff/behavior` to compare the metrics. If a GGUF is not ready,
the controller saves an integrity-only bundle. Do not send a behavior report
or benchmark references in a check request. GSM8K is separate; see
[GSM8K protocol](#gsm8k-protocol).

For conversion, set the local worker's `engine_env` as shown in
[GGUF output](#gguf-output). Private
BF16 conversion also accepts tied Qwen2 models and up to 1100 MiB of input.
This limit does not change ordinary conversion. For checks, build the
[CPU runner](crates/eval/README.md#scratch-behavior-checks). Put the two
pinned dataset files in a local cache. Set these keys in the worker's
`engine_env`:

- `BREWERY_SCRATCH_CHECKS_SCRIPT`: Absolute path of `tools/scratch_checks.py`.
- `BREWERY_SCRATCH_CHECKS_RUNNER`: Use
  `/arboretum/pins/scratch-check-logits-xa-f5abfc6/scratch-check-logits` on the local host.
- `BREWERY_SCRATCH_CHECKS_RUNNER_SHA256`:
  `f5abfc6d29a150f2f08c6366c277bfa2a1a22094b7190b6ba0fce71842ce6465`.
- `BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR`: Absolute path of the directory
  with the four libraries listed in [CPU trace jobs](#cpu-trace-jobs). Use
  `/arboretum/pins/scratch-check-logits-xa-f5abfc6/lib` on the local host.
- `BREWERY_SCRATCH_CHECKS_CACHE`: Absolute path of the local dataset cache.

Before you install these settings, run `sha256sum -c SHA256SUMS` in the pin
directory. The worker does not download data during a check. Restart the
controller after a change to the local worker's runtime configuration.

### Registered evaluation sets

Use `POST /api/admin/evaluation-sets` to register an immutable set. This
operation is not available through MCP. Set `BREWERY_EVALUATION_OPERATOR_FILE`
to a private file owned by the controller account (mode 600). Put `operator`
and `token` in the JSON file. Use a lower-case operator ID and a random
64-digit lower-case hex token. Restart the controller. Send the token in the
`Authorization: Bearer` header. Do not put the token in a saved prompt.

1. Review both sources and their licenses. Save each exact file as
   `<sha256>.<format>` in a private source directory. Do not put this directory
   under a shared model root or in `BREWERY_CONTRAST_CACHE`.
2. Prepare a spec with `name`, `positive`, `negative`, `sizes`, and
   `classifier`, as in the [API contract](api/openapi.json). Give each source
   an HTTPS URL with a full 40-digit commit in the path, the byte size, SHA-256,
   license, format, top-level prompt field and literal system message. The
   optional filter selects rows by a field and exact value. Use strings for
   CSV filters. Give the `tune` and `final_gate` sizes per side, from 1 to 32.
3. On the worker, run
   `python3 tools/registered_evaluation.py spec.json sources eval-cache > register.json`.
   The builder does not use the network. It checks the source sizes and hashes,
   groups prompts by case-folded text with normalized spaces, and removes
   groups that occur on both sides. It sorts the groups by hash and assigns
   disjoint rows to the two partitions. It writes one digest-named file per
   partition and adds their pins to the registration body. Send that body to
   the admin route. The controller records the configured operator and time.
   To change a source, classifier, or split, register a new name.
4. Put the cache at an absolute path in the worker's
   `BREWERY_REGISTERED_EVALUATION_CACHE`. Keep it separate from the contrast
   cache. Use the current `scratch_checks.py` and its adjacent
   `registered_evaluation.py`. Build the current CPU runner and set its SHA-256
   as in [Scratch behavior checks](crates/eval/README.md#scratch-behavior-checks).
   The older `--behavior-v1` runner pin above does not support marker checks.

Call `list_evaluation_sets` or `GET /api/evaluation-sets` to read set names
and sizes. For a GGUF pair, add
`"registered_marker":{"name":"<set-name>","partition":"tune"}` to `run_checks`.
Use `final_gate` only once you select the candidate. Fitting and ablation
search cannot load either partition. Compiled-program and HGN checks do not
accept this option.

The first class whose case-insensitive prefix or substring matches wins. Put
exclusion classes before classes that count when they could both match. The
worker generates at most `max_tokens` per reply (1 to 128). In the check
bundle, `registered_marker` gives parent and child rates, per-class counts
and flags for each side. It saves no full replies. Only replies longer than
120 characters can have a 120-character prefix. The existing `benign_refusal`
check does not change. Registered marker scores are not release-qualified.

### Heretic jobs

Heretic 2.1.0 (upstream commit `4f89482c0d63f2552a582215939910f9f45a3e5f`,
AGPL-3.0-or-later) runs as a separate program. Brewery does not vendor its
code. On a Linux x86-64 CUDA worker, make a private mode-700 runtime directory
with at least 20 GiB free. Install `uv`, Python 3.12 and Bubblewrap. Run
`bash tools/heretic/setup.sh PRIVATE_RUNTIME_DIR`. The installer uses a
hash-pinned uv lock and checks the four upstream TOML files. After setup,
`--offline` verifies or restores from the local uv cache. No job installs
packages or accesses the network.

Set `engine_env.BREWERY_HERETIC_RUNTIME` to the absolute runtime directory in
the local worker config. Enable the local SafeTensors editor. Set
`BREWERY_ALLOW_GPU_LOAD` to `1` in `engine_env` and set `admission` to the
local GPU queue wrapper. The wrapper must acquire the shared GPU before it
runs the engine, accept `--wait -- COMMAND`, and preserve the working
directory, environment and standard streams. Do not use a no-op wrapper.

The worker installs the upstream profiles `noslop`, `nosyco`, `terse`, and
`default`. A preset does not authorize its datasets. Supply only reviewed
offline prompt lists: place newline-delimited UTF-8 prompts in
`prompts/<sha256>.txt`. Use `prompts/index.json` to map a dataset label to
`{"sha256":"<digest>"}`. An operator TOML in `configs/` can instead set
`brewery_sha256` next to `dataset`. Use explicit prompt tables for the built-in
scorers and modifier. Do not supply plugin code. Do not use unlicensed upstream
datasets. For a short `noslop` run, use an operator copy of that preset with
pinned Dolly creative-writing prompts and a smaller response budget.

Send `POST /api/scratch/heretic` or call `run_heretic`. Set `parent_id`,
`base_sha256`, `profile` (TOML name without its extension), `trials` (1 to 64),
`trial_index` (zero-based on Heretic's sorted Pareto front), and `seed`.
Use one standalone SafeTensors file with its model configuration and tokenizer
in the same approved directory, or pass a Library `assets_id` from the same
worker. The worker saves merged weights locally. It
does not upload, chat or run Heretic's benchmarks.

Send `POST /api/scratch/heretic_reproduce` or call `reproduce_heretic` with
`parent_id`, `base_sha256` and the community v4 `manifest` object. The base must
have a verified Brewery Hugging Face single-file download receipt for the
same repository, commit and SafeTensors SHA-256. Supply its reviewed prompt
files locally. An environment or export hash mismatch fails the job.

Read the job result to get `scratch_id`. Scratch provenance includes the
Brewery journal with the exact config, seed, trial index, versions, hashes and
base identity. The W3 delta restores the original parent. To measure the
change, make a byte-identical scratch copy of the Library parent. Convert that
copy and the Heretic child to private GGUF with the same assets. Run W4 checks
on the child with the A2 prompt-set pins. Read the checks and compare them with
the parent before you request publication. These results are not
release-qualified.

### Behavior direction checks

Set `BREWERY_CONTRAST_CACHE` to an absolute path in the worker's `engine_env`.
Use a runner built with `--behavior-v1` support. Put each normalized
`brewery.contrast-prompts.v1` set in `<sha256>.json` in this cache.
Use a held-out set with no fit or search-screen prompts when you confirm an
edit. Set `behavior_direction` in
`POST /api/scratch/{id}/checks` or `run_checks`, with `prompt_sets` that
contains the `slop`, `sycophancy`, and `response_length` digests. Keep each
contrast set to 1 to 32 held-out rows. The worker checks each file's hash
before it reads rows. It does not use the training rows.

To measure over-refusal, also add the `benign_refusal` digest from
`tools/direction_prompt_cache.py`. This set contains 16 reviewed standalone
instructions from [Databricks Dolly 15K](
https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a)
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The
script selects a subset with no prompts in the direction-job splits. It
does not vendor the prompts. Share the derived set under the same license
with this attribution. The set has no training split and cannot be used
with `compute_direction`.

The CPU runner uses the model's GGUF chat template and greedy decoding with
a limit of 64 new tokens. The check counts refusal prefixes, stock phrasing,
and replies that start with the expected agreement label. It reports mean
response length in tokens, not characters. It scores the positive side of
the contrast sets. Agreement on prompts that ask for agreement does not
measure unsolicited agreement. `benign_refusal_rate` is the fraction of
benign prompts with a refusal flag. It does not measure compliance with
harmful requests. `refusal_prefixes` can replace
the default prefixes. A trusted local Python check script can supply a
classifier to `main(refusal_classifier=...)`. Return a Boolean to override
the prefix detector, or `None` to use the detector. The request cannot select
or install a classifier. An operator wrapper can load that entry point with
`runpy.run_path` and the trusted absolute path to `tools/scratch_checks.py`.

On the held-out short-answer side of the length set, the worker computes
full-vocabulary `D_KL(parent || child)` in nats. It reports a mean for the
first token and a separate mean over up to four more positions on the same
greedy parent continuation. `kl_continuation_positions` records how many
positions contribute to the second mean. A value of zero means there were
no continuation positions.

Read `behavior_direction` in the check bundle. The worker saves aggregate
scores and one sample pair per contrast scorer, limited to 160 UTF-8 bytes
per reply. For the benign refusal set, it saves only the scores and
`benign_refusal_flags`, in prompt cache row order. It does not save the
replies or produce a refusal direction. W5 behavior diffs include the
numeric scores. These are research results, not release-qualified. Get
operator approval before publication.

### Abliteration workflow through MCP

1. Pin the Library SafeTensors file and the conversion `assets_id`. Use
   `create_scratch` to copy the parent, then `convert_scratch_gguf` to make
   a private baseline. Poll queued jobs with `get_job`.
2. For a native run, register the exact baseline GGUF bytes in the Library
   and verify the digest. Call `compute_direction` on that GGUF. Use its
   job ID and `version: 2` with `get_direction`. The selected layer and
   held-out Cohen's d measure separation, not a change in replies.
3. To probe the direction without changing weights, use
   `create_direction_intervention` and pass the saved version to
   `start_trace`. Compare with an unmodified trace. Use `ablate_direction`
   for one weight edit or `search_ablation` for a bounded search. Search
   evaluates the selected scratch children and compares the screen with
   matched random controls. Use the child IDs and check jobs from the
   search receipt.
4. For a Heretic run, call `run_heretic` on the original Library SafeTensors
   ID, not the scratch copy. Use `reproduce_heretic` only for a community v4
   manifest with a verified matching base. Read `scratch_id` from `get_job`
   and inspect the child with `get_scratch`. The Brewery journal is
   provenance; it is not the community v4 input. Reinstall the pinned
   launcher with `bash tools/heretic/setup.sh PRIVATE_RUNTIME_DIR --offline`
   when you upgrade to the launcher that records per-trial scores.
5. Convert each child with the same assets as the baseline if needed.
   Use `run_checks` with the exact child hash and
   `behavior_direction.prompt_sets`. Use the same disjoint holdout pins for
   all children in a comparison. Use `get_checks` to read W4 locality and
   perplexity, first-token and continuation KL, marker rates and paired
   flags. Add the optional `benign_refusal` pin to measure over-refusal;
   there is no native refusal direction. Do not treat a check that completes
   as proof of an improvement.
6. Open **Abliteration** for direction pins, native screen and paired
   results, or the Heretic config, trials and journal. Follow the child link
   to **Edits** to see the behavior-direction scores. The workbench views
   do not submit jobs. Only the operator can approve a publish request.

On Qwen2.5-0.5B-Instruct, the tested native directions carried only a
small part of the sycophancy signal. Neither A3 nor A5 found significant
paired flips. Stock slop was too rare to measure a reliable reduction.
Heretic `noslop` did not show a measurable reduction either. These
experiments are not release-qualified.

### Review and approve

Review `check_bundle` for the exact `sha256`. If smoke or behavior did not
pass, put the current bundle `id` in `acknowledge_bundle` to confirm that you
reviewed the incomplete results. A new check replaces the bundle and clears
its acknowledgement. The controller checks the file hash again before it
accepts the publish request. You cannot change the bundle after the request.
Discard and recreate a pending scratch revision if it has no bundle.

**Operator only:** Approve a pending request only after you review the
selected revision, its SHA-256, and its check bundle. Do not expose the
approval endpoint in agent tools. The endpoint uses the same network and
origin checks as Settings.
The controller has no separate operator identity. A trusted client with
direct workbench access on loopback, LAN, or tailnet can call this endpoint
without using the UI. This is an accepted risk. Give direct access only to
trusted clients.

Approval checks the selected file hash and registers one normal SafeTensors
Library model. The result is **not release-qualified**. Approval does not
deploy the model or change release checks. Normal worker source-access rules
still apply to the approved file. An unapproved scratch ID is not a
conversion source. Use the existing conversion workflow after approval if
the worker can read the file and supports the source and target formats.

#### Token loss diff

Build and pin the current CPU check runner as described in
[Scratch behavior checks](crates/eval/README.md#scratch-behavior-checks).
Run scratch checks again after you update the runner. Old bundles and the
old runner can still report aggregate perplexity, but they do not have
token scores. The check bundle records the SHA-256 of the private score
file. The controller checks that hash on each read.

In **Edits**, select the child and click **Token loss diff**. The first
sentence has the largest absolute change. Select a different sentence to
see its tokens. The view shows parent and child log probabilities and ranks
the tokens by absolute loss change. A negative loss change means the child
gives the token a higher probability.

Use `GET /api/diff/token-loss?a=<parent>&b=<child>` or MCP
`read_token_loss_diff`. Both accept `sentence`, `start` and `end`. These
indices start at zero. `end` is exclusive. A slice contains up to 128
predicted tokens and excludes the prompt. The response also includes each
sentence's contribution and the top eight token contributors for the
selected sentence. The share divides the sentence's gain by the signed
net gain over the heldout split. It can be negative or greater than 100%.
If the net gain is zero, the share is null. The result is not
release-qualified.

#### Findings

In the workbench, open **Findings** to read the hypothesis, experiment
timeline, verdict, and links to the evidence. Each write creates a new
version. Use the returned version ID for the next write. A finished
version does not change. The list shows the 100 newest versions.

1. Use `create_finding` (`POST /api/findings`) before the experiments.
   Enter `hypothesis`, `model_id`, and `model_sha256`. For a scratch
   model, enter its revision hash, not its private GGUF hash.
2. After a job succeeds, use `add_evidence`
   (`POST /api/findings/{id}/evidence`). For circuit, trace, or sensitivity,
   set `evidence` to `{"kind":"job","job_id":"…","paths":["/clean_metric"]}`
   and select paths from the job result. For token loss, use
   `{"kind":"token_loss","a":"…","b":"…","paths":["/total_logprob_gain"]}`.
   Match the original `sentence`, `start`, and `end` arguments if used.
   For an imported crosscoder, use `kind: crosscoder` and `artifact_id`.
   The controller reads the artifact and extracts the numbers. Supply a
   JSON Pointer to each number. Do not submit numbers as evidence.
3. Use `add_evidence` with `verdict` to record `supported`, `refuted`, or
   `inconclusive` and give the reasoning. You can send `evidence` and
   `verdict` in the same call. A new experiment without a verdict resets
   the verdict to inconclusive.
4. Use `replay_finding` (`POST /api/findings/{id}/replay`) with the ID and
   SHA-256 of a different model. For a scratch target, convert it first.
   Use `get_finding` (`GET /api/findings/{id}`) to poll the new version.
   Use `list_findings` (`GET /api/findings`) to list the findings and their versions.
   When it is ready, compare the recorded numbers side by side. Check the
   runtime and fixture pins before you give the replay a verdict.

For REST, send `Idempotency-Key` with each write. For MCP, send
`idempotency_key`. Use the same key and arguments only for a retry.
External clients supply `client_id`. This label is self-reported.
The built-in agent records its session ID automatically.

Token-loss replay repeats a direct edit or a prune without a calibration
trace. It uses the same conversion assets and runs measured checks on the
new child. Use **Edits** to inspect or discard that child. Imported
crosscoders and traces with model-bound dependencies can be evidence
but cannot be replayed by the controller. Only one replay runs at a time.
Findings and replays are not release-qualified.

#### Compiled programs

Run `sh tools/provision_compiled_program.sh fetch` to fetch the pinned CPU
compiler sources and Python wheels. Use `offline` in place of `fetch` to
sync the environment from the local cache. The environment uses Python
3.12.15 under `/arboretum/tmp/brewery-w13`; it does not modify the system
Python. Tracr and ALTA both use the Apache-2.0 license. The ALTA compiler uses
its pinned source tree; its training and data packages are not required.

Set these controller environment variables before startup:

- `BREWERY_COMPILED_PROGRAM_PYTHON`: `/arboretum/tmp/brewery-w13/.venv/bin/python`.
- `BREWERY_COMPILED_PROGRAM_SCRIPT`: Absolute path of `tools/compiled_program.py`.

Enable a local SafeTensors editor. Use MCP `compile_program` with `worker`
and `program`, or `POST /api/scratch/compile_program` with the same fields.
The available programs are `sort`, `histogram`, `reverse`, `frac_prevs`,
`alta_parity_relative`, and `alta_parity_absolute`. The endpoint accepts only
these names. It does not accept code or paths.

Use MCP `run_program` with `id`, `sha256`, and `tokens`, or send `sha256`
and `tokens` to `POST /api/scratch/{id}/run_program`. Use one to four
integer tokens. Use tokens `0` to `3` for Tracr and `0` to `1` for ALTA.
`frac_prevs` returns the fraction of `1` tokens up to and including the
current position. Both ALTA programs return parity at each position. Each
run loads the compiled weights. It does not save the inputs or output.

Use the same scratch check and publish routes. The compiled revision is a
root, not an edit of a model. Its parent ID is the nil UUID and
`parent_sha256` is the SHA-256 of zero bytes. It has no `delta.bin`. The
compiler stores `config.json` with `model.safetensors`. The provenance
records the compiler commit, program name, config hash, and runner hash.
The local CPU runner checks every valid input sequence and compares the
decoded outputs with the program specification. Review
`program_correctness` and `program_cases` in the check bundle. You cannot
waive a failure. Language-model behavior checks do not apply; acknowledge
the bundle before you request publication. An operator still must approve.

These models do not support GGUF, llama.cpp traces, or edit deltas. They run
only in the CPU reference runner. Brewery does not inject a compiled
program into the residual stream of another model.

To use the tools without the controller, make an empty artifact directory
and use the pinned Python:

```sh
PYTHON=/arboretum/tmp/brewery-w13/.venv/bin/python
"$PYTHON" tools/compiled_program.py compile sort "$artifact"
"$PYTHON" tools/compiled_program.py run "$artifact" '[2,0,2,1]'
"$PYTHON" tools/compiled_program.py check "$artifact"
```

The compile result includes the SHA-256 of the weights and config. The same
program produces the same hashes on the verified platform with the pinned
environment.

#### Compiled-program benchmark

Use `interp_benchmark` with no arguments or `GET /api/interp/benchmark`.
The controller returns the saved [report](qualification/interp_benchmark.v1.json).
It does not run a model or train features. The results are not release-qualified.

Use the local reference runner to capture a compiled model:

```sh
"$PYTHON" tools/compiled_program.py trace "$artifact" '[2,0,2,1]' "$trace"
```

Choose a new output directory. `manifest.json` uses trace schema v1.
The `l_out`, `attn_out`, and `ffn_out` tensors contain the residual after
each layer, the total attention output, and the MLP output, respectively.
The `compiled_heads` extension contains projected head outputs in
`[head][position][residual coordinate]` order. Positions include the compiler
BOS when the program has one. Use these captures in the local benchmark.
Do not submit them to the Lanternfish job or pinned GGUF import routes.

The compiler writes `ground-truth.v1.json` with the weights and config.
It records the ordered residual basis and the layer, role, head index
where applicable, and residual coordinates for each operation. Tracr
selectors are implemented by the attention heads of their consumers.
ALTA uses the same shared attention and transition rules at every layer.
The sidecar binds the map to the weight and config hashes. The local
compile command returns the sidecar hash. The controller saves the hash
in provenance and checks the sidecar before running a program.

**X5 circuit ranking** (CPU reference runner):

| Program | Method | Evaluation | Precision at K | Recall at K | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| Sort | `attribution_patching` | Ranking | 0.700 | 0.700 | 0.700 |
| Sort | `activation_patching` | Ranking | 1.000 | 1.000 | 1.000 |
| Histogram | `attribution_patching` | Recall-only | Not applicable | 1.000 | Not applicable |
| Histogram | `activation_patching` | Recall-only | Not applicable | 1.000 | Not applicable |
| Histogram with one unused head | `attribution_patching` | Ranking | 0.750 | 0.750 | 0.750 |
| Histogram with one unused head | `activation_patching` | Ranking | 1.000 | 1.000 | 1.000 |
| ALTA relative parity | `attribution_patching` | Recall-only | Not applicable | 1.000 | Not applicable |
| ALTA relative parity | `activation_patching` | Recall-only | Not applicable | 1.000 | Not applicable |

X5 uses exact JAX automatic differentiation through the W13 reference
runner, not the Lanternfish GGUF gradient path. It computes gradients
on the corrupt input for the clean-versus-corrupt logit margin at the
last output position where the decoded outputs differ. Linear attribution
ranks each head and MLP by the sum of the absolute attribution across
tokens. Exact activation patching replaces each site's corrupt output
with its clean output in a separate run and ranks by the absolute
measured margin change.

**Linear attribution misses saturated sites.** In Tracr sort, the
MLPs at layers 0 and 1 and attention head 0 at layer 2 each had an
attribution score of zero. Patching any one of these sites increased
the margin by approximately 2.0. Tracr's sharp nonlinearities can make
the local gradient zero even when replacing an activation changes the
result. This is why X5 confirms selected candidates with real patches.
Patching only the selected candidates cannot recover sites that the
gradient ranking excludes.

K is the number of compiler sites. At the cutoff, count tied sites
fractionally: if three of six tied sites are true and three slots
remain, count 1.5 true sites. ROC AUC gives half credit to a tied
positive-negative pair. Histogram and relative parity have no negative
sites. Their recall at K is automatic and does not measure ranking
quality. The local benchmark control (`histogram_padded`) gives the same
outputs and adds one zero-weight head as a known negative. It tests
ranking with an unused site, not with an active competing circuit.

**W11 TopK SAE** (mean best decoder cosine to the compiler subspace):

| Program | Mean cosine |
| --- | ---: |
| Sort | 0.834 |
| Histogram | 0.935 |
| ALTA relative parity | 0.716 |

**X6 paired crosscoder** (same cosine measure for each decoder):

| Pair | Sort | Descending sort |
| --- | ---: | ---: |
| Sort vs. descending sort | 0.858 | 0.869 |

SAEs and crosscoders use the same residual rows as the trace exporter.
They split full-length inputs 75:25 before training and exclude BOS rows.
For each operation, they compare decoder vectors active during training
to the operation's compiler subspace after the layer that writes it.
For ALTA, they measure final-layer transition state, not attention
read ports cleared by the MLP. These cosines measure alignment, not
whether a feature causes the program output.

To reproduce the report, use a separate copy of the pinned W13
environment with `torch==2.7.1+cpu` from `https://download.pytorch.org/whl/cpu`.
Choose a new working directory outside controller data. The script
writes the compiled models, truth maps, v1 traces, and feature weights
there, and writes the report to standard output:

```sh
"$PYTHON" -B tools/interp_benchmark.py build "$work" > "$report"
```

#### Scratch files

The controller stores these files in its private
`BREWERY_DATA/scratch/{id}/` directory:

- `model.safetensors`: Full child checkpoint, with the original header and
  all unedited bytes.
- `model.gguf`: Private BF16 output, if conversion completed. The scratch
  record contains its hash and converter version.
- `provenance.json`: `brewery.scratch-provenance.v1`. It contains the parent
  and child IDs and SHA-256 hashes, worker ID, timestamp, tool and runtime
  versions, method, and the names, shapes, and dtypes of affected tensors.
  A direct edit records its ordered operations. A merge records its
  algorithm, parameters, and the ordered input IDs, hashes, and weights.
- `behavior.v1.json`: A file for comparisons with the immediate parent.
  It contains the parent and child IDs and the measured metrics.
  The metrics list is empty if the bundle has no behavior report.
- `delta.bin`: Reversible XOR delta (`brewery.weight-delta.v1`). It starts
  with ASCII `BWDELTA1`, then two 64-byte lowercase ASCII SHA-256 hashes
  (parent first), the full file size (`u64`), and a record count (`u32`).
  Each record contains an absolute file offset (`u64`), a nonzero byte
  count (`u64`), and that many XOR bytes. Records do not overlap and are in
  offset order. All integers are little-endian. A copy has no records.

To reconstruct the parent, verify the child hash, then XOR each record into
a separate copy of the child. The result must match the parent hash. Never
apply the delta in place. The HTTP API returns the summary, not the file.

## Limits and behavior on failure

- The worker library reads at most 16 MiB of JSON from one source. The
  evaluator rejects job input larger than 256 KiB.
- The worker is configured with `source_roots` to limit the directories that
  model paths may use. The controller rejects model paths that are not
  absolute, or that contain `..` or control characters.
- The evaluator does not start a Halogen GPU load unless the operator sets
  `BREWERY_ALLOW_GPU_LOAD=1` on that worker. `halogen-proof` also refuses to
  start when `MemAvailable` is smaller than the declared bundle files plus
  12 GiB. This check does not reserve memory. Do not stop a service to free
  memory unless you have approval.
- When an action is not possible, the web page shows the reason from the host
  record.
- If the SSH link fails, the controller keeps the job in `running`. Poll
  the state until the link returns.
- If the controller restarts while a job runs, it logs the interruption and
  checks the same job ID on its assigned worker. It does not send the job to
  another host. An unreachable worker is not evidence that its job stopped.
- Partial outputs are not published. A conversion writes to a new directory
  and never overwrites an old one.
- A result that uses a development subset is not release evidence. The
  `synthetic_fixture` option of `brewery-eval` exists only for tests. Its
  results never count for release.
- Register canonical paths to regular files and directories. The worker
  rejects symlinks. A Hugging Face snapshot composed of symlinks to a blob
  cache is not a directly usable worker source. Materialize a separate copy
  on a volume with sufficient space, without modifying the original.

## Release qualification

### Requirements

A release needs all of these:

1. **Trace parity.** One dense GGUF and one MoE GGUF replay the same token IDs
   through the same Lanternfish build revision on CPU, AMD gfx1151, NVIDIA
   sm_89, and RTX 5070 Ti (sm_120). The comparison covers signal coverage,
   layers, token positions, shapes, routing IDs, and numeric values. Numeric
   values must be within the documented tolerances. The record must show the
   real GPU execution, the cards, the drivers, and the build identities. A
   CPU fallback does not pass.
2. **Halogen identity.** Start an isolated Halogen instance with each profile
   (HGN v2 and HT43). Identify the files that the engine actually opened
   against the selected bundle, without disturbing an existing service. A
   model alias or a successful generation does not prove this.
3. **Library and transfer.** Sources are registered by host and path.
   Originals are never changed or deleted. Transfers preview, resume, and
   verify hashes. Cache cleanup is explicit. Jobs show progress, support
   cancellation and reconnect, and are marked interrupted after a restart.
4. **Prompt comparison.** A saved prompt runs against two GGUF models. Each
   output shows the model identity, template, settings, token positions, and
   trace. Trace values are compared directly only when inputs, positions, and
   signal meanings match.
5. **Conversion.** Convert the same full checkpoint to both profiles on each
   of the three target GPUs (six runs). Verify every output through the
   workbench. Keep independent evidence of GPU conversion for all six runs.
6. **Quality.** One qualified Halogen host runs generation and the full
   GSM8K protocol for each distinct bundle. Reuse a result only if the model,
   all sidecars, assets, runtime, and protocol match. For release
   qualification, compare each full-split score with the score from its exact
   BF16 source checkpoint. The candidate must be no more than two percentage
   points lower. Ordinary precision-neutral references support diagnostics
   only. Without matching BF16 source evidence, release quality is not
   established.

All six paths and both profiles must pass. Development subsets never qualify
a release. Use the [operator matrix guide](qualification/release_matrix.md)
to collect the conversion, verification, runtime and evaluation job records.

### Fixed parity tolerances (`trace-parity`, fixed-v1)

| Signal | Absolute | Relative |
| --- | --- | --- |
| Logits | 0.05 | 0.02 |
| Residual, attention and FFN | 0.01 | 0.02 |
| Routing weights and gate | 0.001 | 0.02 |

Routing IDs and their order must match exactly. A value passes when
`|GPU - CPU| <= atol + rtol * |CPU|`. The `--atol` and `--rtol` options are
for diagnosis only and are not valid release evidence.

### Optional CUDA encoder activity

To observe encoder kernels in the converter that the worker starts, build
the collector with the installed CUDA 13.4 CUPTI headers and library. Use
a build slot and an existing, private operator directory:

```sh
g++ -O2 -std=c++17 -fPIC -shared -I/opt/cuda/include \
  -o "$COLLECTOR" qualification/encoder_activity.cpp \
  -L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64 -lcupti
nvidia-smi --query-gpu=uuid,compute_cap --format=csv
```

Set `BREWERY_CUPTI_COLLECTOR` to the absolute path of this trusted `.so`
and `BREWERY_CUDA_GPU_UUID` to the full `GPU-` prefixed UUID in the worker
environment. Keep the collector outside job-writable directories. Do not
override a device restriction imposed by admission. If `CUDA_VISIBLE_DEVICES`
is already set, it must match the UUID exactly; an existing CUDA injection
also prevents this capture.

Only `convert` jobs with a CUDA encoder option request capture. The native
JSON on stdout does not change. Keep `converter-evidence/encoder-activity.json`
from the job with its `encoder_activity` result. `observed` binds completed
encoder counts and the physical GPU capability to the converter PID,
executable hash, and published bundle digest. Decode activity does not count.
`not_observed` or `unavailable` does not fail the conversion, but supplies
no positive hardware evidence. Observation does not prove that every tensor
used the GPU, and does not replace independent review or file verification.

### GSM8K protocol

The protocol ID is `gsm8k-zero-shot-raw-v1`. The full test split of
`openai/grade-school-math` at revision
`3101c7d5072418e28b9008a6636bde82a006892c` is embedded in `crates/eval`.
It has 1,319 rows and SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The evaluator processes every row in dataset order. It fixes the prompt,
temperature at zero, and token limit at 512. Candidate generation uses greedy
decoding. It rejects prompt, token-limit and subset overrides.

The prompt for each row is:

```text
Solve the following math word problem. Show your reasoning and put the final numeric answer on the last line as #### <number>.

Question: <question>
Answer:
```

The evaluator scores the last nonblank line that begins with `####`.
It normalizes signed decimals and valid thousands grouping. Missing or invalid
answers get zero credit. A source results file must hold all per-question
answers, its source SHA-256, and the matching dataset and protocol identities.

Every `evaluate` result includes the candidate `score`, `identity`, and
`gsm8k-evaluation` artifact. Without a reference, `baseline_score`,
`baseline_identity`, `comparison`, and `reference_scope` are null. With a
reference, the result also includes its score and identity, plus per-question
comparison data. Ordinary references have scope `diagnostic`. That comparison
does not qualify a release.

### Qualification commands

Check a bundle on disk:

```sh
cargo run -p halogen-proof -- qualify /path/to/bundle.json
```

Prove which files an isolated Halogen engine maps. This loads a full model
on the GPU, so run it only on a quiet host with enough free memory, under
the host GPU and build gates:

```sh
cargo run -p halogen-proof -- prove /path/to/bundle.json "$HALOGEN_IMAGE_ID" 900 --allow-gpu-load
```

See `crates/halogen-proof/README.md` for the manifest format, memory budget,
and cleanup.

Compare one CPU trace and one GPU trace:

```sh
trace-parity [--require-moe] [--require-shared-gate] CPU.json GPU.json
```

The exit code is 0 for a pass, 1 for invalid or divergent evidence, and 2 for
a usage error. A pairwise pass does not identify the device. Record the
device, driver, and build identity separately.

The model pins are in `qualification/replay-inputs.json`. Run the replay on
one worker, under that worker's GPU admission lock, with
`qualification/replay_host.py`:

```sh
python3 qualification/replay_host.py --dense DENSE.gguf --moe MOE.gguf \
  --out NEW_RUN_DIR --device DEVICE --lanternfish PATH --comparator PATH
```

The run directory must not exist. Do not reuse one. The script checks the
size and SHA-256 of both models first. If the first comparison fails, the
runner still captures and compares the second model. It writes `failed.json`
and exits nonzero if either comparison fails. Only two successful comparisons
produce `result.json`. A model load or capture error stops the run.

`qualification/reference.md` describes a separate derived-FP32 experiment.
Do not use its results in the native gate.

### Status (2026-11-18)

**Not released.** Evidence from the project record:

- **Software.** Workspace tests cover the controller, worker, native engine,
  evaluator, trace comparator and Halogen identity checks. The HTTP smoke
  test passed using a loopback SSH worker, and the controller ran a real
  two-model GGUF prompt comparison. These checks do not qualify full models.
- **RTX 5070 Ti (sm_120).** Dense and MoE CPU-versus-GPU traces ran on the
  GPU. Both fail the fixed-v1 numeric thresholds at the first logit
  (differences 0.597267 for dense and 1.158570 for MoE). For the MoE model,
  35 of 120 layer and position rows had different expert order. An
  independent llama.cpp run without capture reproduced the first-token
  difference. For the Granite layer-0 fixture, the project diagnosis found
  no arithmetic defect. The tolerances were not changed.
- **NVIDIA sm_89.** A Granite (MoE) replay ran on the worker with full
  offload after an admission-lock repair. It failed fixed parity at the
  first logit. The dense model was not run.
- **AMD gfx1151.** Untested, not failed. It needs a HIP build of the pinned
  llama.cpp revision and an approved build and GPU window.
- **Halogen identity.** The isolated proof tool exists. No full Halogen
  load is proven. The inspected host had too little free RAM for a second
  full instance.
- **Conversion and GSM8K.** Bounded v2/HT43 fixtures on sm_120 matched the
  CPU outputs and showed CUDA encoder kernels. The full checkpoint has not
  been converted on all six paths. Scoring tests on the full GSM8K split
  used synthetic predictions, not measurements of HGN model quality.

Until all requirements pass, treat every result from this workbench as
diagnostic.

## Smoke test

`qualification/workbench_smoke.py` starts the controller on a free loopback
port with private temporary state. It uses two local worker hosts and checks
these items through HTTP: the web page loads, model and prompt registration,
transfer preview and transfer, restart persistence, and that unregistering a
model leaves the original file unchanged.

```sh
python3 qualification/workbench_smoke.py --bin-dir target/release --web-dist web/dist
```

Add `--ssh` to exercise a temporary, key-pinned SSH server on loopback.
Add two `--gguf` options and set `BREWERY_LANTERNFISH` to also run a real CPU
comparison. The smoke test is not release evidence.

## Managed benchmarks

A `benchmark` job runs fixed benchmark adapters on a worker. The adapters
call a model endpoint on that worker host. The feature is off until the
operator turns it on for each host. A benchmark result never qualifies a
release. See [Release qualification](#release-qualification).

### Enable the capability

1. Install the adapters (see [Pinned adapters](#pinned-adapters)).
2. Write the benchmark config file on the worker (see
   [Benchmark config file](#benchmark-config-file)).
3. In the worker file, set `engine_env.BREWERY_BENCHMARK_CONFIG` to the
   absolute path of that file:

   ```json
   {
     "root": "/var/lib/model-brewery/worker",
     "source_roots": ["/home/me/models"],
     "engine": "/home/me/model-brewery/target/release/brewery-engine",
     "engine_env": { "BREWERY_BENCHMARK_CONFIG": "/etc/model-brewery/benchmarks.json" }
   }
   ```

4. In the dispatch file, opt in the host. List the targets and adapters that
   the web page can offer. Each entry is `{"id": "...", "label": "..."}`. The
   `label` is optional.

   ```json
   "workstation": {
     "local": true,
     "worker": "/home/me/model-brewery/target/release/brewery-worker",
     "config": "/home/me/.config/model-brewery/worker.json",
     "capabilities": { "benchmark": true },
     "benchmark_targets": [{ "id": "local-model", "label": "Local model, port 8080" }],
     "benchmarks": [
       { "id": "tool-eval-bench", "label": "Tool eval, mock tools" },
       { "id": "betterbench", "label": "BetterBench decode, chat" }
     ]
   }
   ```

The capability stays off if `capabilities.benchmark` is not `true`, or if
`benchmark_targets` or `benchmarks` is empty. An ID uses 1 to 64 characters:
lowercase letters, digits, `-`, and `_`. It starts with a letter or digit.
IDs in one list are unique. The controller does not read the worker config
file. Make the IDs in the dispatch file match the IDs in the worker config
file. If an ID is missing from the worker config, the job fails.

### Benchmark config file

The file is at most 128 KiB. Unknown fields are errors.

```json
{
  "targets": {
    "local-model": { "endpoint": "http://127.0.0.1:8080/v1", "model": "model-name-as-the-endpoint-reports-it" }
  },
  "adapters": {
    "tool-eval-bench": {
      "executable": "/opt/brewery-bench/tool-eval-bench",
      "version": "3.0.0",
      "protocol": "tool-eval-safe-v1"
    },
    "betterbench": {
      "executable": "/opt/brewery-bench/betterbench",
      "version": "0.6.1",
      "protocol": "betterbench-decode-chat-v1"
    }
  }
}
```

- `targets.<id>.endpoint` is an `http` or `https` URL with a numeric
  loopback host (`127.x.x.x` or `[::1]`) and no credentials, query, or
  fragment. A host name such as `localhost` is refused. The managed adapters
  also accept only the path `/` or `/v1`. The endpoint must not need an API
  key. The worker host must run the endpoint.
- `targets.<id>.model` is the model name that the endpoint expects.
- `adapters.<id>.executable` is an absolute path. The engine follows
  symlinks and runs the target file without arguments.
- `adapters.<id>.version` and `protocol` are labels. They are copied into the
  report. The managed adapters refuse a request with any other label.
- The adapter ID is the benchmark name. The two managed adapters accept only
  the IDs `tool-eval-bench` and `betterbench`.

### Pinned adapters

The `benchmarks/` directory holds two adapters. Each adapter runs one pinned
upstream tool through a fixed recipe. The adapters check the installed
version and refuse any other version.

| Adapter ID | Upstream (verified pin) | Protocol ID | What it sends | Metrics |
| --- | --- | --- | --- | --- |
| `tool-eval-bench` | tool-eval-bench 3.0.0, commit `d85126ee2a812e8acaf046ec8c0ad0446b9ebbbe` | `tool-eval-safe-v1` | Scenarios TC-01, TC-02, and TC-04 with mock tools only. | `mock_tool_subset_score` (%, higher is better). |
| `betterbench` | GGZ14/BetterBench 0.6.1, commit `f72bb9011a830fd98e82ab89d01e549efa76e9c5` | `betterbench-decode-chat-v1` | One warmup and 4 measured streaming chat requests, greedy. | `ttft_p50` (ms, lower), `decode_tps_p50` (tokens/s, higher). |

`tool-eval-safe-v1` scores only measured model outcomes. A wrong tool call
is a low score. An upstream error, an excluded scenario, or a missing
scenario fails the run. `betterbench-decode-chat-v1` fails the run if one
measured request fails, a stream has fewer than two updates, or the endpoint
does not report token usage.

The adapters need Linux and Python 3.11 or later. Install both upstream
tools in one virtual environment that only a trusted user can write:

```sh
python3 -m venv /opt/brewery-bench/venv
/opt/brewery-bench/venv/bin/pip install \
  'tool-eval-bench==3.0.0' 'httpx==0.28.1' 'python-dotenv==1.2.4' \
  'git+https://github.com/GGZ14/BetterBench@f72bb9011a830fd98e82ab89d01e549efa76e9c5'
```

The engine starts an adapter with `PATH=/usr/bin:/bin`. The adapter script
starts with `#!/usr/bin/env -S python3 -EBs`, so it would use the system
Python, not the virtual environment. For each adapter, make a small wrapper
script that starts the pinned Python by absolute path. Use the wrapper as
`executable`:

```sh
#!/bin/sh
exec /opt/brewery-bench/venv/bin/python -EBs /home/me/model-brewery/benchmarks/tool-eval-bench
```

For `betterbench`, change the last path to
`/home/me/model-brewery/benchmarks/betterbench`. Set the mode of each wrapper
to `0755`. Do not add arguments: the adapters refuse them.

The adapters, wrappers, and virtual environment are trusted code. They run
as the worker user and are not a sandbox. Do not let other users write to
them. The report records the SHA-256 of the executable file that the config
names (the wrapper). It does not cover the virtual environment or the
adapter script. Pin those separately. The engine fails a run if the
executable changes while the run is active.

### Run a benchmark

In the web page, open **Benchmarks**, select a host, a target, and 1 to 8
suites, then select **Queue benchmark run**. For the HTTP API:

```sh
curl -s -X POST http://127.0.0.1:4141/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"benchmark","host":"workstation","input_ids":[],
       "options":{"benchmark_target":"local-model","benchmarks":["tool-eval-bench","betterbench"]}}'
```

A benchmark job has no inputs or prompt, and `options` has only those two
keys. The engine runs the suites one after the other, in the order given.
Each suite has a limit of 300 seconds. The controller returns `409` with the
host reason if the capability is off. It returns `400` if the target or
suite ID is not in the dispatch file.

### Write your own adapter

An adapter is any executable file with an absolute path in the config. It
does not need the pinned tools. The engine does this for each suite:

1. Start the file with no arguments. The working directory is a new empty
   directory. The environment is only `PATH=/usr/bin:/bin`, and `HOME` and
   `TMPDIR` are set to that directory. Stderr is discarded.
2. Send one JSON object on stdin. All fields are strings:

   ```json
   {"benchmark":"id","target":"local-model","endpoint":"http://127.0.0.1:8080/v1",
    "model":"model-name","version":"3.0.0","protocol":"my-protocol-v1"}
   ```

3. Read one JSON object from stdout (at most 64 KiB) and exit with code 0:

   ```json
   {"metrics":[{"name":"accuracy","value":0.75,"unit":"ratio","direction":"higher"}],"samples":4}
   ```

   `metrics` has 1 to 64 entries. Each `name` is unique (at most 96
   characters). `unit` has at most 32 characters. `value` is a finite
   number. `direction` is `higher`, `lower`, or `neutral`. `samples` is
   optional. No other field is allowed.
4. Optional: write `raw.json` (valid JSON, a regular file, at most 4 MiB)
   in the working directory. The engine saves it as an artifact. It deletes
   every other file.

The engine stops the whole process group after the adapter exits or after
300 seconds. A file that an adapter writes cannot exceed 4 MiB.

### Read the results

The job result has the schema `brewery-benchmark-report-v1`. It has `target`,
`model`, `status`, `runs`, and `release_qualified` (always `false`). The
engine saves it as `benchmark-report.json`.

- `status` is `complete` if every suite succeeded, `partial` if some did,
  and `failed` if none did.
- Each entry in `runs` has `benchmark`, `version`, `protocol`, `status`
  (`succeeded` or `failed`), `metrics`, `duration_ms`, and
  `executable_sha256`. It can also have `samples`, `error`, and
  `raw_artifact`. A failed suite keeps its `error` and has no metrics.
- Failures are per suite. A failed suite does not stop the next suite.

The **Benchmarks** page shows each suite and a side-by-side table. **Export
JSON** in the scorecard panel saves `brewery-scorecard-<time>.json` with the
schema `brewery-workbench-scorecard-v1`. For each selected job, it holds the
job ID, host, request, state, and the unchanged report. It also lists
`failed_benchmarks` (suites that did not succeed) and `missing_benchmarks`
(requested suites that have no run). Use these two lists to check coverage.
GSM8K jobs are separate entries and are not matched to benchmark jobs.

The workbench does not add, average, or rank metrics across suites or jobs.
Compare only runs that have the same target, model, version, and protocol.
The export always has `release_qualified: false`. A good benchmark score
does not show that a model, a conversion, or a release is correct.

## Web settings

Settings has two sections for operational policy.
Infrastructure also manages host admission and resource policy.
It can install a pinned worker bundle on a selected SSH host after confirmation.
It does not accept arbitrary commands.

| Section | What it does |
| --- | --- |
| **Worker settings** (`/api/settings`) | Saves benchmark target profiles for defined hosts, including installed SSH workers and the backend download service. Workers with a usable evaluation policy can also apply targets, approved suites, baseline endpoints and the evaluation switch. |
| **Infrastructure** (`/api/admin/*`) | Controls weight editing, job admission and supported resource limits for the selected host. It also manages catalog runtimes and SSH definitions. |

Both sections use the same API request protections.
Infrastructure has no second admin key.
The HTTPS gateway does not ask for a login. Thus each client on an allowed
LAN or tailnet network can use all Infrastructure actions. These include
host policy, resource limits, catalog runtimes, editor directory grants,
SSH host definitions and worker installation. This is an accepted risk.
Allow only trusted client networks.
The loopback listener has no login. For network access, use the HTTPS
gateway and limit access to trusted client networks.

### What the operator can set

Select a worker in **Settings**. If it has no active evaluation policy, add a
benchmark target and save the profile. The controller stores the target for
that worker without testing the endpoint, changing job permissions or starting
an evaluation. When a worker policy becomes available, review the saved targets
before applying them. The policy still controls what can run.

- **Targets.** A benchmark target has an ID, an optional label, an endpoint and
  the model name that the endpoint expects. The endpoint must be exactly
  `http://127.0.0.1:PORT/v1` or `http://[::1]:PORT/v1`. The address is on the
  worker host. To apply the target for execution, the port must be in the
  approved ports of that worker. User
  names, queries, fragments, host names and other paths are refused. The
  endpoint must not need a key, and the API has no field for a secret.
- **Suites.** The operator enables suites from the approved catalog only. The
  catalog lists the adapter programs. The operator cannot add a program.
- **Baseline endpoints.** These are off unless worker policy allows them. Each
  one needs the SHA-256 of the reference model served and a runtime ID, as well
  as an endpoint of the same kind. Optional `precision` is descriptive only.
  Settings cannot mark a reference as BF16-attested.
- **Evaluation.** The switch can be on only if the worker policy allows it,
  `BREWERY_ALLOW_GPU_LOAD` is `1` on the worker and
  `BREWERY_HALOGEN_IMAGE_ID` holds a pinned image. The page cannot turn on
  GPU load.
- **Limits.** At most 32 targets, 32 suites and 32 baselines. An ID uses 1 to
  64 lowercase letters, digits, `-` or `_`, and starts with a letter or
  digit. A label or model name has 1 to 256 bytes and no control characters.
  The request body is at most 64 KiB.

### Save is not ready

A save writes a target profile on the controller or applies settings to a
worker with an evaluation policy. The saved revision grows by one.

- **Saved.** The settings are valid for the worker policy and are on disk.
  Saving starts no job. It does not test any endpoint.
- **Ready.** `ready` is true when the policy, the suite catalog and the saved
  settings can be read and are valid. It does not show that a model server
  listens on the port. A job still fails if the endpoint does not answer.

Saving a controller profile does not change `GET /api/state`. For applied
worker settings, `benchmark` is on when the
worker has at least one target and one suite. `baseline` is on when it has a
baseline endpoint. `generate` and `evaluate` are on when evaluation is on.
Otherwise the host shows the reason. A save adds no event to the event stream.
Read the state again.

To avoid a race, a save needs the `revision` that you read last. If another
save came first, the controller returns `409`. Read the settings again and
decide. A save also returns `409` while the host has a queued or running job,
or a job with host `auto`. The worker checks the settings again when a job
starts.

### Which workers can save evaluation settings

Every defined host has its own Infrastructure settings, including the backend
download host and installed SSH hosts. Select **Weight editing settings** or
**Admission and resource limits** for that host. You do not need an evaluator
to pause a host or configure a supported weight editor.

The following prerequisites apply to worker-side execution, not to saving a
controller profile:

- The evaluation worker is **local** and has an absolute `config` path.
- No dispatch enrollment flag is required to use an eligible worker policy.
- The worker file has a `managed_settings` object with `catalog`,
  `allowed_ports`, `evaluation_allowed` and optional `baseline_allowed`
  (default `false`). `root` must be a private directory (mode `0700`, owned by
  the worker user, no symbolic link in its path).
- The catalog file is at most 128 KiB. It maps each suite ID to an absolute
  `executable`, a `version` and a `protocol`. The executable must be a
  regular executable file. It holds at most 32 suites.
- A worker that pins a source (`pinned_publication` or
  `BREWERY_EXPECTED_SOURCE_SHA256`) cannot apply evaluation settings to its
  qualification runtime.
  This protects its qualification configuration. Two managed evaluation
  workers cannot share or nest their `root` directories.

```json
{
  "root": "/var/lib/model-brewery/eval-worker",
  "source_roots": [],
  "engine": "/opt/model-brewery/bin/brewery-engine",
  "managed_settings": {
    "catalog": "/etc/model-brewery/suites.json",
    "allowed_ports": [8080],
    "evaluation_allowed": false
  }
}
```

`deploy/lan/local_eval.py` stages such a worker in a new directory. It copies
the binaries and the adapters, and it writes a proposed dispatch file next to
them. It does not change your live dispatch file. You review and install that
file yourself.

### Infrastructure

Settings > Infrastructure manages policy for defined hosts.
It also has runtime catalog and SSH host controls.
Host policy covers dispatch hosts and installed SSH hosts.
It also covers the built-in `backend-hf` service when that service is configured.
Catalog runtime actions affect only workers listed in `BREWERY_ADMIN_CATALOG`.

The page reads controller state and submits changes.
The controller applies admission rules.
The worker enforces the resource limits it receives.

| Action | Effect |
| --- | --- |
| `enroll` | Moves a worker from `unregistered` to `pending`. |
| `approve` | Runs a transport readiness probe, then moves a `pending` or `inactive` worker to `active`. Only an `active` worker is visible to the workbench. |
| `deactivate` | Moves an `active` worker to `inactive`. |
| `enable_runtime`, `disable_runtime` | Turns the capabilities of a catalog runtime on or off for that worker. A disabled runtime shows `Disabled by infrastructure administration`. Enabling needs a pinned digest and the runtime helper. |
| `install_runtime` | Asks the runtime helper to promote a pre-staged runtime. It needs the helper, a `pending` or `inactive` worker and no running job. It does not install a worker or engine. |
| `set_resource_policy` | Allows or blocks use of the worker for host `auto`. It cannot add a worker to a pool that the dispatch file excludes. |
| `set_host_policy` | Saves the pause state and restricted job types for a defined host. It needs the current host `revision`. |
| `set_host_resources` | Replaces the resource overrides for a defined host. Use `{}` to inherit all worker settings. |

The controller rejects new work when its destination or any input host is
paused. It does not cancel work that already runs. Existing work can finish.
A restricted job type must be a configured capability. Removing a restriction
does not add a capability.

The `hosts` rows in `/api/admin/state` report `installed_jobs`, `allowed_jobs`,
`resources`, `resource_bounds` and `effective_resources`. `installed_jobs`
lists job types from host configuration. `allowed_jobs` lists job types that
remain permitted after capability and job restrictions. The separate `paused`
field controls new work. `resources` shows saved overrides.
`resource_bounds` reports allowed ranges. `effective_resources` shows the
controller's scheduling values, not live resource use.

| Resource | Effect |
| --- | --- |
| `concurrency` | Limits concurrent conversion jobs on a conversion-pool host. The default is `1`. |
| `cpu_threads` | Caps the CPU threads used by an engine job. The worker can apply a stricter local limit. |
| `min_free_ram_bytes` | Sets the minimum free RAM required before a compatible job starts. |
| `disk_reserve_bytes` | Sets the free disk space to keep for supported jobs, transfers and worker caches. |

The controller uses `concurrency` when it schedules pool conversions.
The worker applies its local CPU limits and checks free RAM and disk before
supported jobs start. RAM and disk settings are admission checks, not live
quotas. Configure compatible `resource_bounds` in the dispatch entry and
worker file. `backend-hf` supports host pause and job restrictions, but it has
no adjustable resource bounds. Its disk reserve stays fixed.

Host policy and runtime actions appear in the audit table in
`$BREWERY_DATA/admin.db`. The state shows the latest 50 entries. SSH host
definitions use a separate private registry, so their actions do not appear in
this audit table.

Host policy and resource changes need the latest `revision`. A stale revision
returns `409`. An invalid job type or resource value returns `400`. An unknown
host returns `404`. The API keeps the existing request protections and has no
idempotency key.

#### SSH hosts and user-level worker installation

The SSH host form stores host ID, display name, address, user, port and host-key
fingerprint in the private controller registry under
`$BREWERY_DATA/ssh-hosts`. It writes a private dispatch overlay. It does not
edit the operator dispatch file or `~/.ssh/config`.

In **Add SSH host**, enter the address and port, then probe the host key. The
controller requests the Ed25519 public key from that endpoint without logging
in. Check the displayed SHA-256 fingerprint against the host console or a
trusted message from the administrator: a first connection alone does not
prove the host's identity. Accept the key to use it in the form, or enter a
known fingerprint manually. Neither probing nor accepting saves the host.
The form does not accept a password or private key.

The SSH login key identifies the controller to the remote account. It is
separate from the host key fingerprint. In **Add SSH host** or **Edit
definition**, select a login key from the controller account's `~/.ssh/`
directory, or enter the absolute path to a private key on the controller.
Save the definition before testing it. Saving does not try the key, change
the server fingerprint, or authorize the public key on the remote host.

1. Save the host definition. This does not connect to the host.
2. Select **Test connection**. The helper checks the fingerprint and tests
   key-based SSH login to a non-root Linux account.
3. Select **Plan installation**. The helper checks the pinned bundle and
   remote architecture, then returns the worker paths and a plan ID.
   Planning does not install files on the host.
4. Review the plan. Confirm it in the form before you select **Install on
   host**.

The helper installs `brewery-worker`, `brewery-engine` and `worker.json` under
`~/.local/share/model-brewery/workers/<host-id>/<manifest-sha256>/`. The
configuration sets `source_roots` to `[]`. The helper uses a pinned manifest
and checks both binary hashes. The controller then uses a private SSH
configuration with strict host-key checking.

The SSH workflow does not create accounts, use `sudo`, install system
services, or change firewall, GPU, driver, power or admission settings. It
does not delete remote files. An installed host's SSH identity cannot be
changed; use a new host ID. The controller refuses edits or removal while the
host has queued or running jobs or an active conversion.

#### Configure SSH worker installation

Set these variables in the controller service environment:

1. Set `BREWERY_SSH_PROVISION` to the absolute path of the trusted SSH
   provisioning helper. Use `tools/brewery-provision` from this repository.
2. Set `BREWERY_PROVISION_BUNDLE` to a directory with `manifest.json`,
   `brewery-worker` and `brewery-engine`.
3. Set `BREWERY_PROVISION_MANIFEST_SHA256` to the SHA-256 digest of the exact
   `manifest.json` file.
4. Optional: set `BREWERY_PROVISION_IDENTITY` to an existing SSH private-key
   file as a service default. A login key saved in a host definition takes
   precedence for that host and does not require a service restart.
5. Restart the controller.

The manifest pins the bundle architecture and each binary hash. The helper
accepts Linux bundles for `x86_64` or `aarch64`. The controller needs
`/usr/bin/python3`, OpenSSH `ssh` and `ssh-keyscan`. The remote host needs
`/usr/bin/python3`. If the helper is unavailable, the page cannot test, plan
or install a worker.

### Configure the infrastructure catalog

1. Set `BREWERY_ADMIN_CATALOG` to the absolute path of a private catalog file
   (at most 128 KiB). Each worker ID must exist in the dispatch file:

   ```json
   {
     "workers": {
       "worker-a": {
         "ssh_alias": "frame",
         "initially_active": false,
         "runtimes": {
           "evaluator": {"label": "Evaluator", "capabilities": ["evaluate"], "sha256": "<64 lowercase hex>"}
         }
       }
     }
   }
   ```

   For a local worker use `{"local": true, "initially_active": true}`. The
   controller then checks that the worker program and its config are trusted
   files. A runtime can only list capabilities that the dispatch entry has.
2. Optional: set `BREWERY_ADMIN_RUNTIME_HELPER` to a trusted, absolute
   executable. The controller calls it only as `install-pinned` or
   `probe-installed WORKER RUNTIME SHA256`. It must reply with only the actual
   digest as `{"sha256":"..."}`. This repository does not include a runtime
   promotion helper.
3. Restart the controller.

### Limits of this feature

- A policy save does not start a job. The SSH workflow installs only the
  pinned worker and engine files. No page installs or starts a model server.
  Start the endpoint yourself on an approved port.
- No benchmark or evaluation result from these settings qualifies a release.
- If the operator did not enrol a worker, the page shows it as not managed.
  That is not an error.
- The README sections [Managed benchmarks](#managed-benchmarks) and the
  worker file remain the manual way to set up benchmarks.
- Automated tests cover host policy, resource checks, multi-host HTTP behavior
  and confirmed SSH installation on a loopback host. They do not show a
  production deployment.

## Offline usage and profile preview

Offline tools produce reports and apply reviewed expert maps. They run on one
host and do not call the controller. The planner does not load a model. The
Calibration page reads reports. It does not change a checkpoint.

### Expert usage

`qualification/expert_usage.py` counts which experts a saved Lanternfish
replay trace selected:

```sh
python3 qualification/expert_usage.py TRACE [TRACE ...] > usage.json
```

- Each `TRACE` is a Lanternfish replay JSON v1 file. Give one or more
  compatible, non-duplicate traces.
- The script has no options other than `--help`. It writes one line of
  `brewery-expert-usage-v1` JSON on stdout.
- `layers` has one entry for each layer. Each entry has a `prefill` and a
  `decode` block with `observed_positions`, `missing_positions`,
  `selections`, and `experts`.
- Each row in `experts` has `expert`, `selections`, `observed_positions`,
  `selection_fraction`, `weight_samples`, `weight_sum`, and `weight_mean`.
- Limits: 64 trace files, 64 MiB and 512 tokens for each trace, and 512 MiB
  and 32,768 expert rows for each pool. The script fails if a limit is
  exceeded. It does not truncate.
- The output has the trace identities, the trace SHA-256, and the prefill and
  decode token counts. It has no input text, token IDs, or logits.

Read the numbers with care:

- A trace can be sparse. `missing_positions` shows where the trace has no
  routing data. A layer with no observed routing has an empty `experts`
  list.
- Rows list only the selected experts. An expert that is absent from the list
  was not selected at the observed positions. A coefficient of zero is a
  recorded value, not missing data.
- The counts and fractions describe one workload. They are not accuracy,
  importance, or probability. They do not show that an expert can be removed
  or quantized more.

### Expert-map planning and offline rewrites

`qualification/expert_plan.py` ranks experts from a usage report or REAP
activation captures. It prints one line of `brewery.expert-map.v1` JSON to
stdout. The map has `source_sha256`, `layers`, top-level `top_k`, and `review`.
The map sets `review.required` to `true` and
`review.release_qualified` to `false`. Each value in `layers` lists old expert
IDs in new slot order. The first ID moves to slot 0. Each ranking method
rejects maps with more than 32,768 layer-expert rows.

Review the map before you apply it. Keep its exact bytes. Each writer checks
the map SHA-256. The source SHA-256 must match the exact input file or tree
digest.
Use `sha256sum` for a single-file GGUF or HGN source. Use
`expert_safetensors.py source-digest` for a SafeTensors source.

#### Frequency ranking

Use a complete `brewery-expert-usage-v1` report from the exact GGUF that
Lanternfish ran:

```sh
python3 qualification/expert_plan.py frequency \
  --usage usage.json \
  --source-sha256 "$SOURCE_SHA256" \
  > expert-map.json
```

Set `SOURCE_SHA256` to the exact source SHA-256 from trusted inventory. If the
source has dense layers, add `--layers` with a comma-separated list of every
routed layer ID. Add `--keep N` to retain `N` experts per layer. Without
`--keep`, the planner reorders all experts. Frequency ranks experts by
selection count. It sorts from highest count to lowest and breaks ties by
ascending old expert ID.

The planner rejects missing routing in mapped layers. Frequency describes
only the captured workload. It does not measure expert importance or quality.

#### REAP ranking

Use separate calibration and held-out activation captures:

```sh
python3 qualification/expert_plan.py reap \
  --calibration calibration.json \
  --held-out held-out.json \
  --source-sha256 "$SOURCE_SHA256" \
  > expert-map.json
```

Both files must use `brewery.expert-activations.v1` and stay within 64 MiB.
They must have the same source SHA-256, observer SHA-256, and routing
dimensions. Set `split` to `calibration` in the first file and `held-out` in
the second. Record every selected expert at each token and routed layer.
`applied_gate` is the actual combination coefficient. `output_l2` is the
expert output's L2 norm before the gate is applied.

Use disjoint token sequences from independent workloads. The planner rejects
exact repeated sequences. It cannot prove that the workloads are independent.
Every expert must be selected at least once in calibration. Calibration
scores set the rank. Held-out scores do not. The score is the mean of
`applied_gate * output_l2` over tokens that selected each expert. A null
held-out score means no selection. It does not mean a measured zero.

Add `--keep N` to retain `N` experts per layer. The planner requires
at least `top_k` experts and no more than the source expert count. REAP
describes routing on the source. It does not predict routing or quality after
pruning.

#### Apply a map

Calculate the SHA-256 of the reviewed map's exact bytes:

```sh
MAP_SHA256=$(sha256sum expert-map.json | cut -d' ' -f1)
```

Do not edit the map after you calculate its SHA-256. Each writer checks this
digest and the source digest. Each writer creates a new output path. None
overwrites the source or an existing destination.

**GGUF.** The writer accepts one unsharded GGUF v3 file with a `qwen2moe` or
`qwen3moe` layout. The map must include every routed layer. Each layer must
retain the same number of experts, and that number must be at least the
native `expert_used_count`. The plan limit is 8 MiB. The writer supports up
to 512 layers, 1,024 experts per layer, 16,384 tensors, and a 32 MiB header.

```sh
python3 tools/expert_gguf.py preview \
  --source "$SOURCE" --plan expert-map.json --plan-sha256 "$MAP_SHA256"
python3 tools/expert_gguf.py apply \
  --source "$SOURCE" --plan expert-map.json --plan-sha256 "$MAP_SHA256" \
  --output "$NEW_GGUF"
python3 tools/expert_gguf.py verify \
  --source "$SOURCE" --plan expert-map.json --plan-sha256 "$MAP_SHA256" \
  --output "$NEW_GGUF"
```

`preview` reports the projected size and does not write a file. `apply`
rewrites packed tensor slices without decoding or requantizing them. It reads
the staged output back and checks its header and tensor bytes before it
publishes the file. `verify` repeats those checks on an existing output. These
checks do not decode the model or measure quality.

**SafeTensors.** The writer accepts a standalone `.safetensors` file with a
separately pinned config, or an indexed Hugging Face directory with its index
and config. The config must declare the unquantized Flash-Next architecture
and one integrated MTP block. A standalone source requires `--config` and
`--config-sha256`; an indexed source must get its config from the source
directory.

```sh
SOURCE_SHA256=$(python3 tools/expert_safetensors.py source-digest \
  --source "$SOURCE_DIR")
python3 tools/expert_safetensors.py apply \
  --source "$SOURCE_DIR" --plan expert-map.json \
  --plan-sha256 "$MAP_SHA256" --output "$NEW_DIR"
```

Use the `source-digest` output as `SOURCE_SHA256` when you build the map. For
an indexed source, it is a tree digest.

For a standalone source, pass the same pinned config to both commands:

```sh
CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)
SOURCE_SHA256=$(python3 tools/expert_safetensors.py source-digest \
  --source "$WEIGHTS" --config "$CONFIG" --config-sha256 "$CONFIG_SHA256")
python3 tools/expert_safetensors.py apply \
  --source "$WEIGHTS" --config "$CONFIG" --config-sha256 "$CONFIG_SHA256" \
  --plan expert-map.json --plan-sha256 "$MAP_SHA256" --output "$NEW_DIR"
```

Use decimal layer keys. Trunk layers are `0` through `N-1`; MTP uses `N`.
The total source file set is limited to 2 TiB, 16,384 files and directories,
and 100,000 tensors. Each header, index, and config is limited to 4 MiB.
Combined index and shard-header metadata is limited to 16 MiB. Expert tensors
must use BF16, F16, or F32. A reorder can map selected layers. A prune must map
every trunk layer and the MTP layer, with one retained expert count across
those layers. The writer copies the source to the new directory. A prune also
updates the config and, for an indexed source, the index size.

**HGN.** The writer accepts one native HGN v2 file, not a model bundle. It
supports canonical fused `layers.N.mlp` expert tensors, including I4R and
Q4CP planes and Q8G64 or Q6G64 rows. The source limit is 32 MiB, with at most
4,096 tensors and 8,388,608 elements per tensor. Pruning requires a top-level
`top_k` in the map. The writer does not change config or load the output in a
runtime.

```sh
python3 tools/expert_hgn.py "$SOURCE" expert-map.json "$NEW_HGN" \
  --reviewed-sha256 "$MAP_SHA256"
```

#### Evidence limits

Fixture tests check map validation, source and map digests, output structure,
and retained bytes. SafeTensors tests also calculate forward values for
synthetic tensors. These checks do not load SafeTensors or HGN outputs in an
inference runtime.

A separate CPU `llama.cpp` probe loaded tiny `qwen2moe` and `qwen3moe` GGUF
fixtures, then decoded reordered and pruned files. Reordered logits matched
exactly in both fixtures. The pruned files produced finite logits for three
decode steps. This probe does not establish production runtime support or
measure pruning quality. None of these checks qualifies a release.

### Profile preview

`profile_preview.py` estimates the size of a custom recipe. It compares two
built-in profiles, `v2` and `ht43`, and borrows tensors from one profile into
the other.

1. Make one conversion plan for each profile with the pinned converter. Use
   the same source and the same other options:

   ```sh
   bf16-to-hgn plan --source "$SOURCE" --profile v2 > v2.json
   bf16-to-hgn plan --source "$SOURCE" --profile ht43 > ht43.json
   ```

2. Write the recipe. This one takes the `ht43` encoding for the
   `hyper_connection` tensors of layer 0 and keeps `v2` for the rest:

   ```json
   {
     "schema_version": 1,
     "base_profile": "v2",
     "overrides": [{ "select": { "layers": [0], "role": "hyper_connection" } }]
   }
   ```

3. Run the preview:

   ```sh
   python3 qualification/profile_preview.py --recipe recipe.json \
     --v2-plan v2.json --ht43-plan ht43.json > preview.json
   ```

Recipe rules:

- `base_profile` is `v2` or `ht43`. Each override takes the whole tensor from
  the other profile.
- `select` has `layers` (a list of layer numbers), `role`
  (`hyper_connection` or `routed_expert`), or both. `role` can also be used
  alone. A separate form selects exact tensors, for example
  `{"tensors": ["layers.0.mlp.experts.down_proj.weight"]}`.
- There is no action or codec field. You cannot choose a new encoding.

The output has the schema `brewery.profile_preview.v1`. It has the resolved
recipe and its SHA-256, a list of `changes` (`before` and `after` encoding,
shape, and payload bytes for each tensor), and a `size` block with
`base_hgn_bytes`, `predicted_hgn_bytes`, and `delta_hgn_bytes`. The size
includes headers, directory, alignment, and companion tensors. It does not
include assets, scratch space, or the disk reserve.

Limits of the preview:

- The preview is an estimate. It has `executable` set to `false` and
  `release_eligible` set to `false`. It does not make a bundle. No converter
  or runtime can use it yet.
- A change moves a whole tensor. The preview cannot select part of a tensor,
  for example one expert.
- A changed tensor can need a companion tensor, such as a rotation. The
  preview borrows the companion together with the tensor and marks it as
  `kind: "companion"` in `changes`. Check these rows.
- The source check is `metadata_only`, and `weight_identity` is
  `unverified`. Make both plans from the same source. The preview does
  not show model quality.

### Halogen draft acceptance

Import saved Halogen **0.18.22 non-streaming response timings** as JSONL,
one final `chat.completion` or `text_completion` response per line. Use a
single known model build and workload per file:

```sh
python3 qualification/halogen_speculation.py --producer-version 0.18.22 \
  responses.jsonl > speculation.json
```

If each line is a benchmark record containing the native response under
`response`, add `--wrapped`. The importer reads the native `timings` fields,
not benchmark scores. It sums draft rounds, proposals and accepted tokens
separately for the head, lookup and grammar. The fraction is total accepted
divided by total drafted; no proposals means no fraction. Missing
speculation data is counted separately. The report contains no conversation
text or response IDs. Keep the original file private.

This adapter does **not** read streaming chunks, server logs or activation
traces. These timings cannot identify expert routes, acceptance by lookahead
depth, or accuracy on reference answers. The declared version and model
alias do not prove which weights ran. For a small example with its text
removed, use `qualification/fixtures/halogen-0.18.22-timing.jsonl`.

### Calibration page

Open the **Calibration** tab of the web page (`#/calibration`). Use
**Report files (JSON)** to select one or more files from your disk. Each file
can be at most 16 MiB. The browser reads the files. It does not upload them
to the controller and does not load a model. Open reports stay in memory when
you go to another workbench page in the same browser tab. Reloading or closing
the tab clears them. When no reports are open, the page shows the commands
that make each report.

The page accepts these schemas:

- `brewery-expert-usage-v1` (from `expert_usage.py`)
- `brewery.expert-map.v1` (from `expert_plan.py`)
- `brewery.profile_preview.v1` (from `profile_preview.py`)
- `brewery-halogen-speculation-v1` (from `halogen_speculation.py`)

The page shows one error line for any other file. For a map, it shows the
source SHA-256, map file SHA-256 when browser crypto is available, ranking
method, and proposed layer order. It does not apply the map. Each report
stands alone. The page shows descriptive usage statistics, size predictions,
draft acceptance counts, and map orders. These are not calibration or quality
results. A preview is not a runnable profile.

Keep the files local. Use only your own traces and plans. Do not upload
production data to the workbench or to another service for this workflow.

## Hugging Face download transport

Select **Library → Download from Hugging Face**. Enter a repository, and
optionally a revision and an exact model filename. The application backend
runs the download. You do not need to enroll or select a worker. Public
models do not need a token. To access a gated public model, set a read token
in **Settings → Hugging Face access token**.

Library downloads use the pinned Rust `hfr` downloader. No transport switch
is required in the UI: the worker uses native Xet for files with validated
Xet metadata and HTTP ranges for other files in the same revision. Failed
Xet reconstruction does not trigger an HTTP fallback. On retry, unfinished
Xet files restart. The worker independently checks each completed file
against the pinned Hub manifest before it adds the model to the Library.
For gated public models, the long-lived Hub token is not sent to the Xet
content service; the downloader uses a short-lived read ticket.

To edit a downloaded GGUF, SafeTensors, or indexed Hugging Face checkpoint,
select **Edit weights** on its Library entry. The backend uses its local CPU
editor. No transfer or second Library registration is necessary. Packed GGUF
edits support Q4_0 and Q8_0.

For **Tune**, upload two separate JSON batches containing `activations` and
`targets`: one for calibration and one for held-out evaluation. Each upload
has a 256 KiB limit. Select both Library entries in the editor. On other
local workers, register both JSON files from that worker's approved source
roots instead. **Apply** creates a new revision in the source format after
review. It does not change the downloaded file.

## Package the local CPU weight runtime

Use the packager from the same source release as the controller, dispatcher,
worker and engine. Prepare a new private release directory with those binaries.
The build host must have `/usr/bin/python3`, NumPy, PyYAML, Bubblewrap and a
trusted `gguf-py` checkout. Record the dependency fingerprint from the approved
build inputs; pass that pinned value at install time:

```sh
/usr/bin/python3 -I -B tools/brewery-weight-runtime.py fingerprint \
  --gguf-python "$GGUF_PY"
/usr/bin/python3 -I -B tools/brewery-weight-runtime.py install \
  --bin-dir "$RELEASE_BIN" --gguf-python "$GGUF_PY" --dependency-sha256 "$APPROVED_SHA256"
```

The install command prints the asset manifest SHA-256. Keep
`brewery-weight-edit`, `brewery-weight-tune`, and `brewery-weight-runtime/`
together beside the controller. The package uses the target system's Python
ABI and shared libraries; build it on a compatible host. The launchers verify
the pinned manifest and import the helpers inside the sandbox before reporting
capabilities. A previous wrapper that cannot answer `--capabilities` is not
accepted as a ready runtime. Installation does not enable any host policy.

Jobs expose their private staged work directory to the helper. For an
operator-verified SafeTensors source, the package requires Bubblewrap with
`--ro-bind-fd`. The launcher reports this support as `proof_source_fds` and
mounts only the checked manifest members read-only. It does not mount the
source directory or its parent. Reviewed apply still checks the full SHA-256
of each source member before it publishes a new output. The sandbox has no
host network or credential store. If Bubblewrap or unprivileged user
namespaces are unavailable, the helper fails closed.

### Set up editing on an existing SSH worker

In **Settings → Infrastructure**, select the SSH worker that owns the Library
entry, then open **Weight editing**. Setup installs a separate CPU editor;
it does not replace the worker or change the planner's SSH key.

1. Select a verified SSH login for the same server. If none is listed, open
   **SSH hosts**, add or edit the definition for that address and port,
   select a login identity, save, and use **Test connection**. Return to
   the worker's editor settings. Use the endpoint shown for that worker.
   If a server has both LAN and Tailscale addresses, a login tested at one
   address is not proof for the other. You can add a second SSH definition
   for the worker's address without changing the first definition.
2. Enter the absolute model directory **on the worker**. The login account
   must be able to read it. Do not enter a file path or a local directory
   from the machine running the browser.
3. Submit setup. The controller checks the host key, installs the pinned
   package in the remote account's home, and checks that the new worker
   responds. Selecting a login or entering a directory does not install
   anything. Editor jobs become available only after verification succeeds.

Existing model-size and source-proof limits still apply. For Tune, register
the two calibration files on the same worker.

Prepare these controller settings before the first setup:

- `BREWERY_REMOTE_EDITOR_PROVISION`: absolute path to the trusted executable
  `tools/brewery-editor-provision`, beside `tools/brewery-provision`.
- `BREWERY_REMOTE_EDITOR_PACKAGE` and `BREWERY_REMOTE_EDITOR_PACKAGE_SHA256`:
  a trusted, pinned, uncompressed tar archive with `brewery-worker`,
  `brewery-engine`, `brewery-weight-edit`, `brewery-weight-tune`, and
  `brewery-weight-runtime/` at the archive root. Use regular files, not
  symlinks; use binaries from this release and the target's architecture
  and Python ABI. The target needs `/usr/bin/python3`, Bubblewrap with
  `--ro-bind-fd`, and working unprivileged user namespaces.
- Optional: `BREWERY_REMOTE_EDITOR_CREDENTIALS` supplies additional
  worker-specific credentials instead of a verified SSH login from Settings.
  It is an absolute path to a private JSON object keyed by credential ID.
  For example:

  ```json
  {
    "planner-editor": {
      "label": "Planner editor provision key",
      "worker": "frame-hy4-plan",
      "user": "brewery-editor",
      "identity": "/path/to/worker-editor-key",
      "fingerprint": "SHA256:<existing SSH server fingerprint>"
    }
  }
  ```

  The `identity` is a private service-readable key file, not a password or
  browser upload. Use a dedicated non-root account with read access to the
  selected model directory and access to its own private home. The key must
  permit the installer and the sidecar RPC; the original forced-command
  planner key cannot do this. A static SSH alias must already have the
  same Ed25519 pin in its known-hosts file and be reachable directly from
  the controller. Managed SSH definitions reuse their verified pin.

Disabling the editor blocks new editor jobs without removing the installed
runtime or changing the planner. Failed setup leaves the previous verified
settings in place. Old, unselected releases do not receive new jobs; retain
them while any allocations or published outputs still reference their files.

## Weight edit tools (fixtures only)

Use `weight_edit.py` and `weight_tune.py` to propose changes to individual
weights. Review the requested and stored values before applying. Apply writes
a new file in the source format; it never overwrites the source. The tools do
not run a model. A lower calibration error does not establish model quality.

Prerequisites: Python 3, NumPy and `gguf-py` from the pinned llama.cpp
checkout. Run them from the repository root. Keep scratch files under
`/arboretum`, not `/tmp`.

| Script | Does |
| --- | --- |
| `tools/weight_edit.py` | `inspect` a weight, `propose` edits from a request file, `apply` a proposal to a new file. |
| `tools/weight_tune.py` | Proposes edits by calibration. It writes the same proposal format. It never applies them. |
| `tools/weight_impact.py` | Compares two GGUF files of the same layout around one coordinate. |

### Input limits

- One little-endian GGUF v3 or native HGN v2 file, at most 32 MiB. This
  path does not edit a sharded checkpoint or model bundle.
- Editable floating-point weights: F32, F16 and BF16. Editable packed weights:
  GGUF `Q4_0` and `Q8_0`, or native HGN `Q8G64`. For these packed types the
  scale stays fixed. The proposal shows the value the stored code will
  represent. Tuning evaluates only representable changes within the update
  limit. Apply preserves unrelated codes and repairs the affected HGN
  payload checksums.
- A recognized but unsupported tensor appears as read-only. For example, HGN
  HT/I4R and their rotation companions cannot be edited this way. The editor
  never replaces a packed tensor with floating-point storage.
- At most 4096 tensors, 8,388,608 elements in a tensor, 256 edits, and a
  128 KiB request or proposal file.
- A coordinate is a JSON list of zero-based indices in row-major order. The
  last index changes fastest.
- The tools refuse non-finite values, out-of-range or duplicate coordinates,
  a wrong source hash, and an edit that rounds back to the old value in the
  tensor type.

### Manual edit

Give the SHA-256 of the source with every call. A wrong hash stops the run.

```sh
SRC=/arboretum/fixture/dense-f32.gguf
SHA=$(sha256sum "$SRC" | cut -d' ' -f1)

# 1. Read one weight.
python3 -B tools/weight_edit.py inspect "$SRC" --source-sha256 "$SHA" \
  --tensor blk.0.attn_q.weight --index '[0,0]'

# 2. Write a request: ops are "set" (new value) or "delta" (add).
echo '[{"tensor":"blk.0.attn_q.weight","index":[0,0],"op":"set","value":4.0}]' > request.json

# 3. Propose. Read the proposal before you apply it.
python3 -B tools/weight_edit.py propose "$SRC" --source-sha256 "$SHA" request.json > proposal.json

# 4. Apply to a NEW file. The destination must not exist.
python3 -B tools/weight_edit.py apply "$SRC" proposal.json /arboretum/fixture/edited.gguf
```

The proposal has `schema` `brewery.weight-edit-proposal.v1`, the
`source_sha256` and a list of `edits`. Each edit holds `tensor`, `index`,
`op`, `value`, `dtype`, and the `before` and `after` values in the tensor
type. `apply` writes one new file, checks all its bytes, and prints
`brewery.weight-edit-result.v1` with `source_sha256`, `output_sha256` and
`edit_count`. It does not overwrite an existing file.

### Automatic tuning

`weight_tune.py` picks new values for the weights you name. You give it
calibration data and held-out data. Each file is JSON with exactly two keys:
`activations` (rows) and `targets` (rows).

```sh
python3 -B tools/weight_tune.py "$SRC" --source-sha256 "$SHA" \
  --tensor blk.0.attn_q.weight --indices '[[0,0]]' \
  --calibration cal.json --held-out held.json --max-update 8 \
  > proposal.json 2> report.json
```

The proposal goes to standard output. A JSON report goes to standard error.
The report gives the mean squared error before and after, on the calibration
data and on the held-out data. It also sets `regression: true` for the
held-out data when the error grew. Read it: a change that helps the
calibration data can hurt the held-out data. The tool does not hide that.

- `--max-update` caps the change of each weight, measured after rounding to
  the tensor type. If no value inside the cap changes the stored weight, the
  run fails.
- The calibration and held-out data must not share an activation row. The
  tool refuses an overlap, also when the targets differ.
- The proposal does not depend on the held-out data or on the order of
  `--indices`.

### Manual and automatic give the same kind of result

Both ways write the same proposal format. Review a proposal in the same way,
then use the same `apply` command. A manual edit and an automatic one that
choose the same coordinate and value give the same proposal edit, and the same
output file. `qualification/weight_revision_smoke.py` checks this on a fixture:
it applies a manual proposal and a tuned proposal, quantizes both to Q8_0, and
compares the reports. It needs a built `llama-quantize` and the synthetic
`dense-f32.gguf` catalog fixture.

### Compare before and after

Quantization works on blocks. An edit to one weight can change its neighbours
in the same block. `weight_impact.py` shows this for two files that came from
the same quantizer and the same recipe:

```sh
python3 -B tools/weight_impact.py before.q8.gguf after.q8.gguf \
  --tensor blk.0.attn_q.weight --index '[0,0]'
```

It prints one JSON object: `selected_before`, `selected_after`, `block_size`,
`changed_neighbors` (changed values in the same block),
`changed_values_outside_block` and `other_changed_tensors`. The two files must have the same header and
layout, at most 32 MiB each, and the tensor must have at most 4,194,304
elements. A non-empty `other_changed_tensors` or a count outside the block
means something else changed.

### What this does not show

- No model runs. There is no inference, no perplexity and no benchmark.
- An edit that lowers a calibration error is not a quality result, and a
  passing check is not a release qualification.
- The tests use small fixtures. Do not use these scripts on a full model, and
  do not put them in a production path.
- An isolated worker can opt in to the `weight_inspect`, `weight_propose`,
  `weight_tune` and `weight_apply` jobs. Submit Library IDs, not file paths.
  Apply requires a reviewed proposal for the same source and host. Only a
  verified apply adds an edited file to the Library. Select **Library → Open
  weight editor**. The deployed `local-gguf` worker supports the GGUF and HGN
  types listed above. It requires at least 16 GiB of available RAM before
  it starts a job.

Tests: `TMPDIR=/arboretum python3 -B -m unittest discover -s qualification -p 'test_weight_*.py'`.
