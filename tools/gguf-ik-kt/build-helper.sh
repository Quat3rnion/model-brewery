#!/usr/bin/env bash
# Run under a build slot: build-gate.sh build bash build-helper.sh ARCHIVE NEW_HELPER
# ARCHIVE must be the codeload tar.gz for ik_llama.cpp at the exact commit below.
set -euo pipefail

commit=7c00f8788e303bfba2886138e288effb0fd5fbbf
digest=a8eb039b318be127f24fcfe9a287eaf3acd64727ee48bf779d0316491a7194cc
if [[ $# != 2 ]]; then
    printf 'usage: %s ARCHIVE NEW_HELPER\narchive: %s\n' "$0" \
        "https://codeload.github.com/ikawrakow/ik_llama.cpp/tar.gz/$commit" >&2
    exit 2
fi

here=$(dirname -- "$(realpath -- "$0")")
destination=$(realpath -e -- "$(dirname -- "$2")")/$(basename -- "$2")
[[ ! -e "$destination" && ! -L "$destination" ]] || { echo "helper destination already exists" >&2; exit 1; }
stage=$(mktemp -d -- "$(dirname -- "$destination")/.ik-kt-build.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp -- "$1" "$stage/source.tar.gz"
printf '%s  %s\n' "$digest" "$stage/source.tar.gz" | sha256sum --check --status \
    || { echo "ik_llama.cpp archive digest mismatch" >&2; exit 1; }
mkdir -- "$stage/source"
tar --extract --gzip --file="$stage/source.tar.gz" --directory="$stage/source" --strip-components=1

cmake -S "$stage/source" -B "$stage/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE -DGIT_EXECUTABLE=/usr/bin/false \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_F16C=ON -DGGML_FMA=ON \
    -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
cmake --build "$stage/build" --target ggml --parallel 2
c++ -std=c++17 -Wall -Wextra -Werror -O2 -static -s -I"$stage/source/ggml/include" \
    "$here/kt_helper.cpp" "$stage/build/ggml/src/libggml.a" -pthread -o "$stage/helper"
[[ $("$stage/helper" identify) == "brewery-ik-kt-v1 $commit 158 153 154 155" ]]
ln -T -- "$stage/helper" "$destination"
sha256sum -- "$destination"
