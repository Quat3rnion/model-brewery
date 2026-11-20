# Install Model Brewery

This guide installs the controller and web page from a Linux clone. It uses a
user-level systemd service, keeps state outside the clone, and adds no worker
hosts by default.

## Get the source

Clone the repository, then run the remaining commands from its root:

```sh
git clone https://github.com/Quat3rnion/model-brewery.git
cd model-brewery
```

## Requirements

- Linux with a working systemd user manager.
- Rust 1.89 or later.
- Node.js 22.12 or later and npm.
- Python 3 for project CLI scripts.
- A C compiler and linker, `pkg-config`, and SQLite development headers.
- Git to identify each installed source revision.
- `curl` for the read-only API checks below.

The first build needs access to the configured Cargo and npm package sources.
Cached packages can allow an offline build. The controller and web page do not
need a GPU, Docker, or SSH access.

To use the optional SSH worker installer later, the controller host needs
`/usr/bin/python3`, `ssh`, and `ssh-keyscan`. The remote Linux host also needs
`/usr/bin/python3`. See [`OPERATOR.md`](OPERATOR.md) before you configure
workers.

## Build

Run these commands from the root of a fresh clone:

```sh
cargo build --workspace --release --locked
npm --prefix web ci
npm --prefix web run build
```

The Rust binaries go to `target/release/`. The web page goes to `web/dist/`.

## Install a release

Install each source revision in its own directory. This keeps a running
release separate from the clone and preserves earlier releases.

Run this block from the clone root:

```sh
set -eu

base="$HOME/.local/share/model-brewery"
release_id="$(git rev-parse --short=12 HEAD)"
release="$base/releases/$release_id"

if [ -e "$release" ] || [ -L "$release" ]; then
  printf 'Release path already exists; inspect it before continuing: %s\n' "$release" >&2
  exit 1
fi

install -d -m 755 "$release/bin" "$release/web"
install -m 755 target/release/brewery "$release/bin/brewery"
install -m 755 target/release/brewery-dispatch "$release/bin/brewery-dispatch"
cp -a web/dist "$release/web/dist"
```

Create the private state directory and select this release for a first
installation:

```sh
set -eu

base="$HOME/.local/share/model-brewery"
release_id="$(git rev-parse --short=12 HEAD)"
release="$base/releases/$release_id"

for path in "$base/state" "$base/current"; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    printf 'Path already exists; inspect it before continuing: %s\n' "$path" >&2
    exit 1
  fi
done

install -d -m 700 "$base/state"
ln -s "$release" "$base/.current-$release_id.new"
mv -T "$base/.current-$release_id.new" "$base/current"
```

The controller requires `BREWERY_DATA` to name a private directory. The
`state` directory uses mode `0700`. It stores the controller database and SSH
host definitions. Keep it when you update or stop the service.

## Install the user service

The service listens only on `127.0.0.1:4141`. The controller has no login.
Do not expose this address on a network interface.

Create the service file. This command stops if a file with the same name
already exists:

```sh
set -eu

unit="$HOME/.config/systemd/user/model-brewery.service"
mkdir -p "$HOME/.config/systemd/user"

if [ -e "$unit" ] || [ -L "$unit" ]; then
  printf 'Service file already exists; inspect it before continuing: %s\n' "$unit" >&2
  exit 1
fi

umask 077
cat > "$unit" <<'EOF'
[Unit]
Description=Model Brewery local controller and web page

[Service]
Type=simple
Environment=BREWERY_LISTEN=127.0.0.1:4141
Environment=BREWERY_DATA=%h/.local/share/model-brewery/state
Environment=BREWERY_WEB_DIST=%h/.local/share/model-brewery/current/web/dist
Environment=BREWERY_DISPATCH=%h/.local/share/model-brewery/current/bin/brewery-dispatch
WorkingDirectory=%h/.local/share/model-brewery/current
ExecStart=%h/.local/share/model-brewery/current/bin/brewery
UMask=0077
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now model-brewery.service
```

Use `systemctl --user` for this service. Do not install a system-wide unit.
The user service starts when your user manager starts. To keep it running
after logout and start it at boot, check linger:

```sh
loginctl show-user "$USER" -p Linger
```

If the result is `Linger=no`, enable linger if your system allows it:

```sh
loginctl enable-linger "$USER"
```

This command can require administrator approval.

## Verify the service and Settings API

Check the service and its recent log:

```sh
systemctl --user status --no-pager model-brewery.service
journalctl --user -u model-brewery.service --no-pager -n 50
```

Open `http://127.0.0.1:4141/` in a browser on the controller host. These
commands read the current API state, saved Settings, SSH host definitions,
and API contract:

```sh
curl --fail --silent --show-error http://127.0.0.1:4141/api/state
curl --fail --silent --show-error http://127.0.0.1:4141/api/settings
curl --fail --silent --show-error http://127.0.0.1:4141/api/admin/ssh-hosts
curl --fail --silent --show-error http://127.0.0.1:4141/api/openapi.json
```

These are `GET` requests. They do not add hosts or change Settings. Use the
Settings page and current API contract before you make an authorized change.

## OpenCode skills

The three project skills are already in `.opencode/skills/`. OpenCode can load
them by ID when it uses this clone:

- `brewery-agent-operator`
- `brewery-calibration-benchmarks`
- `brewery-release-qualification`

You can copy the complete skill directories to your user skill directory for
use outside this clone. The command below keeps any existing skill directory:

```sh
mkdir -p "$HOME/.config/opencode/skills"

for skill in \
  brewery-agent-operator \
  brewery-calibration-benchmarks \
  brewery-release-qualification
do
  destination="$HOME/.config/opencode/skills/$skill"
  if [ -e "$destination" ] || [ -L "$destination" ]; then
    printf 'Keeping existing skill: %s\n' "$destination"
  else
    cp -a ".opencode/skills/$skill" "$HOME/.config/opencode/skills/"
  fi
done
```

## Optional worker hosts

Default to no extra worker hosts. Ask the operator before you define or install
a worker. Do not scan the network, inspect SSH configuration, or enroll hosts
automatically.

If the operator wants a worker, request its host ID, display name, address,
SSH account, and port. In **Add SSH host**, enter the address and port to probe
that endpoint for its Ed25519 host key. Compare the displayed SHA-256
fingerprint with one from the host console or administrator before accepting
it. You can also enter a known fingerprint manually. The probe does not log
in, save a host, or install a worker.

Select a login key from the controller account's `~/.ssh/` directory, or
enter the absolute path to a key elsewhere on the controller. Use a key
whose public key is authorized for the remote SSH account. To change a
defined host, select **Edit definition**. Enter a path, not a password or
the contents of the key.

Use **Settings > Infrastructure > SSH hosts** to define an authorized host.
Saving a definition does not connect to it. Test the connection, review the
installation plan, and install only after the operator confirms that plan.
The installer requires its trusted helper and pinned worker bundle. If either
is unavailable, stop and ask the operator; do not use another enrollment path.

The current routes are `GET /api/admin/ssh-hosts` to list definitions,
`POST /api/admin/ssh-hosts` to save one, and
`POST /api/admin/ssh-hosts/{id}/action` for test, plan, or confirmed install.
Read `/api/openapi.json` before making an API change. Use the UI for the
confirmation step.

## Stop and update

Check the workbench for queued or running jobs before you stop the controller.
A restart can interrupt active work.

Stop the service without deleting state:

```sh
systemctl --user stop model-brewery.service
```

To also stop it from starting with your user manager, disable it:

```sh
systemctl --user disable model-brewery.service
```

Both commands keep the service file, releases, clone, and state directory.

To update, build from the new source revision with the commands in [Build](#build).
Then run only the first command block in [Install a release](#install-a-release).
Do not recreate or replace the state directory.
Use a new commit so the release path is new. Check the path before you switch
the service:

```sh
set -eu

base="$HOME/.local/share/model-brewery"
release_id="$(git rev-parse --short=12 HEAD)"
release="$base/releases/$release_id"
next="$base/.current-$release_id.new"

if [ ! -d "$release" ] || [ -e "$next" ] || [ -L "$next" ]; then
  printf 'Check the new release and link paths before continuing.\n' >&2
  exit 1
fi

ln -s "$release" "$next"
systemctl --user stop model-brewery.service
mv -T "$next" "$base/current"
systemctl --user start model-brewery.service
```

Verify the service again after the update. The old release and state remain in
place. Do not remove either during an update.
