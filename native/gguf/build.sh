#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ $# != 5 || ( $1 != stock && $1 != rocmfpx ) || ! $2 =~ ^(cpu|cuda|hip)$ ]]; then
    echo 'usage: build.sh {stock|rocmfpx} {cpu|cuda|hip} GGML_INCLUDE GGML_LIB_DIR FULL_PIN' >&2
    exit 2
fi
family=$1 backend=$2 include=$3 lib=$4 pin=$5
catalog=../../crates/controller/src/gguf_catalog.json
revision_key=encoder_revision
[[ $family == stock ]] || revision_key=rocmfpx_encoder_revision
[[ $pin =~ ^[0-9a-f]{40}$ && $pin == "$(jq -er --arg key "$revision_key" '.[$key]' "$catalog")" ]] || {
    echo "incorrect catalog pin for $family" >&2
    exit 2
}
lib=$(realpath -e -- "$lib")
mkdir -p .build/tmp
export TMPDIR="$PWD/.build/tmp"
out=".build/$family-$backend"
tmp="$out.new"
rm -f -- "$out" "$out.revision"
trap 'rm -f -- "$tmp"' EXIT
link=()
if [[ -f $lib/libggml.so ]]; then
    link=(-L"$lib" -Wl,-rpath,"$lib" -lggml -lggml-base)
else
    link=(-Wl,--start-group "$lib/libggml.a" "$lib/libggml-base.a" "$lib/libggml-cpu.a" -Wl,--end-group)
fi
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -I"$include" matvec.cpp \
    -o "$tmp" "${link[@]}" -fopenmp -ldl
embedded=$("$tmp" --revision)
if [[ ! $embedded =~ ^[0-9a-f]{7,40}$ || $pin != "$embedded"* ]]; then
    # Tarball builds and builds configured without VCS report "unknown".
    # This attestation must only be used with separately verified sources.
    if [[ $embedded != unknown || ${GGUF_VERIFIED_SOURCE_REVISION:-} != "$pin" ]]; then
        echo "backend revision mismatch: expected $pin, got $embedded" >&2
        exit 2
    fi
fi
printf '%s\t%s\n' "$pin" "$embedded" > "$out.revision"
mv -- "$tmp" "$out"
