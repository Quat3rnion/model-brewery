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
- [Weight edit tools](#weight-edit-tools-fixtures-only)

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
- A Rust toolchain that supports edition 2024 (Rust 1.85 or later).
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
Host and browser origin, and the password before it forwards requests to
the loopback controller. It does not bypass worker permissions or GPU
admission. Without endpoint overrides, the gateway binds to `127.0.0.1`
and accepts only `127.0.0.1` clients and `Host: 127.0.0.1:8443`.

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

Open `https://<configured-LAN-host>:8443/`. Transfer the
public `~/.local/state/model-brewery-lan/tls/lan-ca.crt` to your client over
a trusted channel. Trust that certificate before entering the `brewery`
login. The password is in `~/.local/state/model-brewery-lan/password` on
the host. Do not copy `tls/server.key` or put the password in a URL.
Each logged-in user has controller operator permissions.

For Tailscale, `model-brewery-tailnet.service` requires
`~/.config/model-brewery/tailnet.env`. Set the same four variables for your
tailnet: use this machine's Tailscale addresses for `BREWERY_BIND`, the
allowed tailnet client CIDRs for `BREWERY_CLIENT_NETS`, and the Tailscale IP
or DNS name that clients use for each host value. Open the configured
tailnet host on HTTPS port `8443`. Copy the public certificate
`~/.local/state/model-brewery-lan/tls/tailnet-ca.crt` from the host over
a trusted channel. Trust it on the client, then use the same `brewery`
login. Do not copy any `.key` file. The tailnet gateway does not use
public Funnel access. To remove only tailnet access, run
`systemctl --user disable --now model-brewery-tailnet.service`.

Check the services with `systemctl --user status model-brewery-lan.service
model-brewery-lan-bridge.service model-brewery-lan-standby.service`. Stop
the gateway to remove LAN access without stopping the active qualification
job. The standby waits for the workflow lock; after qualification it serves
the same library and jobs. Stop standby before starting a new
`local_sm120.py` workflow; start standby again after the workflow has
acquired its lock. During controller handoffs the gateway can return `503`.

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

The worker owns its cache. The workbench never changes or deletes an original
model file. Unregistering a library entry removes only the entry.

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
| `BREWERY_LANTERNFISH` | engine | Pinned Lanternfish binary for `compare`. |
| `BREWERY_CONVERTER` | engine | Pinned bf16-to-hgn converter for `inspect`, `plan`, `convert`, and `verify`. |
| `BREWERY_COMPONENT_CONVERTER` | engine | Separate role-overlay converter build for HF jobs that replace a component. See [Conversion components](#conversion-components). |
| `BREWERY_COMPONENT_CPU_ONLY` | engine | Set to `1` to refuse non-CPU backends for those jobs. |
| `BREWERY_CUSTOM_CONVERTER` | engine | Separate CPU-only converter for custom profiles. See [Custom profiles](#custom-profiles). |
| `BREWERY_GGUF_QUANTIZER` | engine | CPU-only `llama-quantize` for GGUF output. See [GGUF output](#gguf-output). |
| `BREWERY_GGUF_HF_EXPORTER` | engine | Optional bounded HF-to-GGUF exporter. |
| `BREWERY_EVAL` | engine | Path of `brewery-eval`. |
| `BREWERY_ALLOW_GPU_LOAD` | eval | Set to `1` to let `generate` and `evaluate` start an isolated Halogen load. Without it, those jobs fail. |
| `BREWERY_HALOGEN_IMAGE_ID` | eval | Full local image ID, `sha256:` and 64 hex digits. Get it from `docker image inspect`. A tag is not accepted. |
| `BREWERY_BASELINE_CONFIG` | eval | Absolute path of the baseline endpoint JSON file. |
| `BREWERY_BASELINE_TOKEN_*` | eval | Bearer token for an endpoint. The endpoint entry names the variable. |

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

A baseline is the score of the exact BF16 source checkpoint. Use a results
file, or name an endpoint that you configure. The endpoint file maps an ID to
an entry:

```json
{
  "bf16-source": {
    "url": "https://baseline.example/v1/completions",
    "model": "model-name-as-the-endpoint-reports-it",
    "source_sha256": "<64 hex digits of the source checkpoint identity>",
    "runtime_id": "runtime-1",
    "bearer_env": "BREWERY_BASELINE_TOKEN_BF16"
  }
}
```

Rules that the evaluator enforces: the URL uses `https`, or `http` on
loopback only. The path is exactly `/v1/completions`. The endpoint must reply
with `X-Brewery-Source-Sha256` and `X-Brewery-Runtime-Id` headers that match
the entry. The evaluator follows no redirects and uses no proxy.

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
  qualification. To check quality, run `generate`, and `evaluate` against a
  `baseline`, as separate jobs.

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

The Convert page says "From the checkpoint, if it has one". It does not know
if the checkpoint has the component. Read `result.components` after a job to
see what was used.

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
| **Runtime accuracy and quality** | Not measured by saving, converting or verifying. | **Unproven.** Run `generate`, `evaluate` and a `baseline` as separate jobs, and compare. Even then it is not a release qualification. |

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
quality. It is not a production worker or a release qualification step.

**Inputs.** `input_ids` is the source, then at most one `imatrix` input.

| Source | Rule |
| --- | --- |
| `gguf` | One file. It must hold only F32, F16 or BF16 weights (and integer tensors). A quantized or split GGUF is refused. The host needs `capabilities.gguf_output`. |
| `hf` | The host needs `gguf_hf_output` as well. The exporter is bounded: it accepts a small, dense Qwen2-style model (at most 512 MiB) with exact tensor names and shapes. See the limits below. |
| `hgn` | Refused. |

**Presets.** `options.preset` is one name from the GGUF catalog
(`GET /api/profiles/gguf-catalog`). The default is `Q4_K_M`. The only other
GGUF option is `threads` (1 to 64). A preset is a whole-model recipe of the
pinned upstream quantizer. The quantizer chooses the type of each tensor.
`decode_backend` and all HGN options are refused. The catalog is pinned to one
encoder revision. It lists only types that the encoder can write. It has no
`IQ1_XXS`, and it has no F32 preset. A name that is not in the catalog is
refused.

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
| `BREWERY_GGUF_QUANTIZER` | engine | Absolute path of a regular, CPU-only `llama-quantize`. The engine runs it offline with a fixed, cleaned environment. |
| `BREWERY_GGUF_HF_EXPORTER` | engine | Optional. The bounded exporter for HF sources. |

A worker that pins a source for release qualification refuses GGUF output.
The API has no route to change these settings.

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

**Hugging Face limits.** The exporter is narrow. The supported Flash-Next
checkpoint (`qwen4_exp`) cannot be exported to GGUF: the exporter reports it
as unsupported (it finds the projector, MTP and engram tensors and refuses).
Do not expect a GGUF from that model by this path. The accepted candidate is an
untied dense Qwen2 model of at most 512 MiB, as a flat indexed or single
`model.safetensors`, with BF16, F16 or F32 tensors, exact expected names and
shapes, a local Qwen2 tokenizer and no extra, quantized or sidecar tensors.
It writes one intermediate BF16 or F16 `model.gguf`, no split. It needs a CPU
PyTorch runtime inside a network-isolated sandbox. The end-to-end runtime check
of the exporter is not yet done. An export does not show that a runtime can
load the file or that its quality is good. For HF sources, `inspect` and
`plan` read metadata and do not promise exact target types; `verify` repeats
the export in a transient file.

### Job kinds

`transfer`, `transfer_preview`, `cache_cleanup`, `compare`, `inspect`,
`plan`, `convert`, `verify`, `generate`, `evaluate`, `baseline`, `benchmark`.

### Model kinds

`gguf`, `hf` (alias `bf16-checkpoint`), `hgn`, `brewery-bundle`, `hgn-v2`,
`ht43`, `gsm8k-baseline`, `gsm8k-predictions`, `assets`, `imatrix`, `ngram`,
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
| `evaluate` | One HGN bundle directory. Add a `gsm8k-baseline` file or `options.baseline_source`, not both. You cannot change the prompts or the token limit. |
| `baseline` | A `gsm8k-baseline` or `gsm8k-predictions` file, or `options.baseline_source`. |
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

Agents can operate the workbench without driving the browser. Start with
[`AGENTS.md`](AGENTS.md) for the project skills. The running controller
serves its [OpenAPI contract](api/openapi.json) at `GET /api/openapi.json`.
Use the same loopback address and worker permissions as the web UI.

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

Settings and administration routes are in
[Web settings and administration](#web-settings-and-administration).

Other routes: `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`,
`GET /api/conversions`, `GET /api/conversions/{id}`,
`POST /api/conversions/{id}/cancel`, and `DELETE /api/models/{id}`. Errors
use `{"error": "..."}`.

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
   all sidecars, assets, runtime, and protocol match. Each artifact must
   score no more than two percentage points below its source. Without a
   matching baseline, quality stays unvalidated.

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

The protocol ID is `gsm8k-zero-shot-raw-v1`. The test split of
`openai/grade-school-math` at revision
`3101c7d5072418e28b9008a6636bde82a006892c` is embedded in `crates/eval`. The
token limit is 512. Greedy decoding is used. A source results file must hold
per-question answers and the matching protocol and checkpoint.

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

Settings has two sections for changing controller or worker policy.
Neither section provisions workers, changes SSH trust, edits a dispatch file
or runs arbitrary commands.

| Section | What it does |
| --- | --- |
| **Evaluation settings** (`/api/settings`) | Saves benchmark targets, benchmark suites, baseline endpoints and the evaluation switch of an enrolled worker. |
| **Infrastructure** (`/api/admin/*`) | Enrols and approves catalog workers, enables or disables their runtimes, and adjusts pool membership within the configured limit. |

Both sections require normal workbench access through the authenticated
gateway or the loopback listener. They use the same request protections.

### What the operator can set

The operator can set only what the **worker policy** allows. The policy comes
from the worker file and from files that the infrastructure owner controls.
The Settings page cannot widen it.

- **Targets.** A benchmark target has an ID, an optional label, an endpoint and
  the model name that the endpoint expects. The endpoint must be exactly
  `http://127.0.0.1:PORT/v1` or `http://[::1]:PORT/v1`. The address is on the
  worker host. The port must be in the approved ports of that worker. User
  names, queries, fragments, host names and other paths are refused. The
  endpoint must not need a key, and the API has no field for a secret.
- **Suites.** The operator enables suites from the approved catalog only. The
  catalog lists the adapter programs. The operator cannot add a program.
- **Baseline endpoints.** These are off unless worker policy allows them. Each
  one needs the SHA-256 of the exact BF16 source checkpoint and a runtime ID,
  as well as an endpoint of the same kind.
- **Evaluation.** The switch can be on only if the worker policy allows it,
  `BREWERY_ALLOW_GPU_LOAD` is `1` on the worker and
  `BREWERY_HALOGEN_IMAGE_ID` holds a pinned image. The page cannot turn on
  GPU load.
- **Limits.** At most 32 targets, 32 suites and 32 baselines. An ID uses 1 to
  64 lowercase letters, digits, `-` or `_`, and starts with a letter or
  digit. A label or model name has 1 to 256 bytes and no control characters.
  The request body is at most 64 KiB.

### Save is not ready

A save writes the settings to the private state directory of the worker. It
does this before the answer. The saved revision grows by one. Two things are
separate:

- **Saved.** The settings are valid for the worker policy and are on disk.
  Saving starts no job. It does not test any endpoint.
- **Ready.** `ready` is true when the policy, the suite catalog and the saved
  settings can be read and are valid. It does not show that a model server
  listens on the port. A job still fails if the endpoint does not answer.

After a save, `GET /api/state` shows the effect. `benchmark` is on when the
worker has at least one target and one suite. `baseline` is on when it has a
baseline endpoint. `generate` and `evaluate` are on when evaluation is on.
Otherwise the host shows the reason. A save adds no event to the event stream.
Read the state again.

To avoid a race, a save needs the `revision` that you read last. If another
save came first, the controller returns `409`. Read the settings again and
decide. A save also returns `409` while the host has a queued or running job,
or a job with host `auto`. The worker checks the settings again when a job
starts.

### Which workers can be managed

The operator of the infrastructure enrols a worker. The page cannot do this.

- It is a **local** worker with an absolute `config` path. An SSH worker
  cannot be managed. A worker that is not enrolled shows `Worker settings
  are locked`. An SSH worker that is marked managed shows `Worker policy or
  suite catalog unavailable`.
- The dispatch entry has `"managed_settings": true`. The controller reads the
  dispatch file when it starts, so restart the controller after a change.
- The worker file has a `managed_settings` object with `catalog`,
  `allowed_ports`, `evaluation_allowed` and optional `baseline_allowed`
  (default `false`). `root` must be a private directory (mode `0700`, owned by
  the worker user, no symbolic link in its path).
- The catalog file is at most 128 KiB. It maps each suite ID to an absolute
  `executable`, a `version` and a `protocol`. The executable must be a
  regular executable file. It holds at most 32 suites.
- A worker that pins a source (`pinned_publication` or
  `BREWERY_EXPECTED_SOURCE_SHA256`) cannot be managed. Its settings stay
  `Worker policy or suite catalog unavailable`. Two managed workers cannot
  share or nest their `root` directories.

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

Settings > Infrastructure and `/api/admin/*` manage the policy that the
controller applies to catalog workers.

| Action | Effect |
| --- | --- |
| `enroll` | Moves a worker from `unregistered` to `pending`. |
| `approve` | Runs a transport readiness probe, then moves a `pending` or `inactive` worker to `active`. Only an `active` worker is visible to the workbench. |
| `deactivate` | Moves an `active` worker to `inactive`. |
| `enable_runtime`, `disable_runtime` | Turns the capabilities of a catalog runtime on or off for that worker. A disabled runtime shows `Disabled by infrastructure administration`. Enabling needs a pinned digest and the runtime helper. |
| `install_runtime` | Asks the runtime helper to promote a pre-staged runtime. It needs the helper, a `pending` or `inactive` worker and no running job. Without the helper, the answer is `409`. |
| `set_resource_policy` | Stops or allows use of the worker for host `auto`. It can lower the pool setting of the dispatch file. It cannot raise it. |

The API **does not**: install a worker or its engine, create or change an SSH
alias, pin or change a host key, edit `authorized_keys`, a worker file or the
dispatch file, change GPU admission, power or reservations, or run commands.
These stay with the operator of the infrastructure. The controller runs one
action at a time and writes applied actions to an audit table in
`$BREWERY_DATA/admin.db`. The state shows the latest 50 entries.

Status codes: `200` applied, `400` bad or unknown field, `403` invalid host
or origin, `404` unknown worker or runtime, `409` action unavailable or check
failed, `503` no catalog workers are configured or another action is running.
A repeated lifecycle action can return `409`. There is no idempotency key.

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
   digest as `{"sha256":"..."}`. This repository has no production helper.
3. Restart the controller.

### Limits of this feature

- A save is not a start. No page starts, stops or installs a model server.
  Start the endpoint yourself on an approved port.
- No benchmark or evaluation result from these settings qualifies a release.
- If the operator did not enrol a worker, the page shows it as not managed.
  That is not an error.
- The README sections [Managed benchmarks](#managed-benchmarks) and the
  worker file remain the manual way to set up benchmarks.

## Offline usage and profile preview

Two offline scripts help you plan a selective quantization recipe. They run
on one host. They do not load a model and do not call the controller. The
Calibration page shows their output. These tools produce reports. They do not
convert a model.

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
to the controller and does not load a model.

The page accepts these schemas:

- `brewery-expert-usage-v1` (from `expert_usage.py`)
- `brewery.profile_preview.v1` (from `profile_preview.py`)
- `brewery-halogen-speculation-v1` (from `halogen_speculation.py`)

The page shows one error line for any other file. Each report stands alone.
The page shows descriptive usage statistics, size predictions, and draft
acceptance counts. These are not calibration or quality results. A preview is
not a runnable profile.

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

To edit a downloaded SafeTensors model, select **Edit weights** on its Library
entry. The backend uses its local CPU editor; no transfer or second Library
registration is needed. **Apply** creates a new revision after review. It
does not change the downloaded file.

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
