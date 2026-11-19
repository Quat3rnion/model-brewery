# Halogen engine load identity

`qualify` checks an HGN bundle on disk. `prove` launches a separate local
Halogen container and checks which files the engine mapped. Neither command
checks output quality or proves that the engine used every tensor.

## Bundle manifest

Use `hgn-v2` or `ht43`. Declare one `checkpoint`, and declare each external
`lookup` and `vision` file you want to qualify. Use paths relative to the
manifest, and use expected SHA-256 digests from a trusted source. Keep the
files immutable for the whole operation. For example, this is a manifest
template; substitute the two expected digests before use:

```json
{
  "format": "ht43",
  "files": [
    {"role":"checkpoint","path":"model.hgn","sha256":"<expected checkpoint SHA-256>"},
    {"role":"lookup","path":"lookup.hgn","sha256":"<expected lookup SHA-256>"}
  ]
}
```

Both formats use HGN schema version 2. The preflight checks the header model
ID for each role, file bounds, and SHA-256. It rejects unknown or repeated
roles, aliases, and symlinks. It does not validate all tensor records. An
external draft, legacy quality overlay, tokenizer, and other undeclared
sidecars are not in scope: do not use engine proof as proof of an API bundle
that needs those files.

## Run

```sh
cargo run -p halogen-proof -- qualify /path/to/bundle.json
cargo run -p halogen-proof -- prove /path/to/bundle.json "$HALOGEN_IMAGE_ID" 900 --allow-gpu-load
```

Use a trusted Halogen image already installed on the same Linux host. Set
`HALOGEN_IMAGE_ID` to the full `sha256:...` ID from `docker image inspect`,
not a tag or registry manifest digest. The runtime interface comes from
Halogen's [engine entrypoint](https://github.com/peonist-ai/halogen-flash-server/blob/a033099786c2915b25d49b6051b4c06e60cbcdaa/deploy/entrypoint.sh):
`engine` mode starts `/usr/local/bin/flash_serve` with `--ck`, `--bind`,
`--port`, `--slots`, `--ctx`, `--max-tok` and `--kv-pool`. The entrypoint's
watchdog defines the engine `PING\n` → `PONG\n` exchange. The CLI uses the
same installed launch interface. It does not call an existing API, publish a
port, start the API frontend, or download models.

Run `prove` under the host's GPU and quiet-build gates, only when there is
enough spare unified memory. It needs access to the local Docker socket,
the supported AMD GPU (`gfx1151`), `/dev/kfd`, `/dev/dri`, and an image with
Python 3 and its normal entrypoint. The preflight refuses to start when
`MemAvailable` is smaller than the sum of declared files plus 12 GiB;
passing this check does not reserve RAM or make a busy host safe. Budget
approximately 110 GiB for stock v2 plus external lookup, or 102 GiB for stock
HT43 plus external lookup, **before** the extra headroom. The image and
container are not started by `qualify`; a local Docker image lookup never
pulls a replacement.

The container has no external network, private IPC and read-only mounts
for the individual files. The checker requires one engine descended from
the container entrypoint and requires that engine to own the private
loopback listener. Before and after a successful `PING`, Linux must report
that the same engine maps the entire declared files at their expected
container paths and device/inode identities, read-only. Unknown `.hgn`
mappings, an unreadable `/proc`, a file changed during the observation, or
a sidecar read but not kept mapped all prevent success. `PONG` or an API
model label alone does not prove load identity. This checks a trusted
runtime against a specific bundle; it is not a remote attestation of an
untrusted Docker daemon or image.

Only successful observation and cleanup produce proof JSON on stdout.
Errors use stderr and a nonzero exit status. A private in-container
watchdog stops the engine if the caller stops renewing its lease for
20 seconds. If the caller is killed before cleanup, a stopped container
may remain. Its ID is printed to stderr; verify that exact ID before
removing it. Never stop a service to free memory without authorization.

## Verification status

CPU fixtures exercise header/digest qualification and the live `/proc`
and socket probe. The workload fixture is not Halogen and does not produce
real Halogen proof. Frame has the v2 and HT43 files, a `gfx1151` GPU, and
local Halogen images (0.19.1 and 0.20.0), but only about 16 GiB of available
RAM at discovery. No full model was loaded and no existing service was
changed. A gated real load on a quiet host remains to be verified.
