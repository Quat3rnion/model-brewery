#!/bin/sh
# Fetch only when requested. Build or rebuild the virtualenv without network access.
set -eu
mode=${1:-}
case "$mode" in
    fetch|offline) ;;
    *) echo "Usage: sh $0 fetch|offline" >&2; exit 2 ;;
esac
cd "$(dirname "$0")"
root=/arboretum/tmp/brewery-w13
export TMPDIR=/arboretum/tmp UV_CACHE_DIR="$root/uv-cache"
mkdir -p "$root"
source_pin() {
    project=$1 pin=$2
    if [ "$mode" = fetch ] && [ ! -d "$root/$project" ]; then
        git init -q "$root/$project"
        git -C "$root/$project" fetch --depth=1 "https://github.com/google-deepmind/$project.git" "$pin"
        git -C "$root/$project" checkout -q --detach FETCH_HEAD
    fi
    test "$(git -C "$root/$project" rev-parse HEAD)" = "$pin"
    test -z "$(git -C "$root/$project" status --porcelain --untracked-files=all)"
}
source_pin tracr 9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd
source_pin alta afb0559da1dc08a3819c5ae20e0547686052b75a
if [ "$mode" = fetch ]; then
    uv python install 3.12.15
else
    export UV_OFFLINE=1
fi
if [ ! -x "$root/.venv/bin/python" ]; then
    uv venv --python 3.12.15 --no-python-downloads "$root/.venv"
fi
uv pip sync --only-binary=:all: --python "$root/.venv/bin/python" compiled_program.requirements.txt
"$root/.venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 12, 15)'
