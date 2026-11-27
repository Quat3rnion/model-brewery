#!/usr/bin/env bash
set -euo pipefail
umask 077
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
root=$(realpath -e -- "${1:?Usage: bash setup.sh PRIVATE_RUNTIME_DIR [--offline]}")
case ${2:-} in
    '') sync_args=() ;;
    --offline) sync_args=(--offline) ;;
    *) exit 2 ;;
esac
[[ -d $root && $(stat -c '%a:%u' "$root") == "700:$(id -u)" ]]
mkdir -p -- "$root"/{cache,tmp,prompts,configs,runs}
export UV_PROJECT_ENVIRONMENT="$root/venv" UV_CACHE_DIR="$root/cache" TMPDIR="$root/tmp"
uv sync --project "$here" --python 3.12 --locked --no-install-project "${sync_args[@]}"
install -m 600 -- "$here/uv.lock" "$root/uv.lock"
install -m 600 -- "$here/run.py" "$root/run.py"
commit=4f89482c0d63f2552a582215939910f9f45a3e5f
while read -r sha preset; do
    path="$root/configs/$preset.toml"
    if [[ ${2:-} != --offline ]]; then
        curl --proto '=https' --fail --silent --show-error --location \
            "https://raw.githubusercontent.com/p-e-w/heretic/$commit/config.$preset.toml" \
            -o "$path.new"
        printf '%s  %s\n' "$sha" "$path.new" | sha256sum -c --
        mv -- "$path.new" "$path"
    fi
    printf '%s  %s\n' "$sha" "$path" | sha256sum -c --
done <<'PINS'
7c251932a17020997ce247e2c32c76162d194edd08e005a97ca9fa15852ae841 default
f58b61c2d7a09c0edd50f4b801411c732e8e9d48726573d22b57bbd2f29fd1a1 noslop
98b6097c005185583446a45a74c6d946e545c3fd24960efcdd744f9d8815a4ad nosyco
c740a6834ff97f64a89c4eccd4c54448b6161d14f6b747c62b6b418888419f4c terse
PINS
"$root/venv/bin/python" -IB "$root/run.py" identity "$root" > "$root/identity.json.new"
mv -- "$root/identity.json.new" "$root/identity.json"
