#!/usr/bin/env bash
# Bounded GLQ fixture checks; CPU by default, GPU test runs are opt-in.
# Regenerate the fixture with: python3 -B glq_oracle.py PINNED_SOURCE glq_vectors.bin
set -euo pipefail
cd "$(dirname "$0")"

target=${1:-cpu}
case "$target" in
    cpu|sm_120|sm_89|gfx1036|gfx1151) ;;
    *)
        echo "usage: bash glq_test.sh [cpu|sm_120|sm_89|gfx1036|gfx1151]" >&2
        exit 2
        ;;
esac
sdk=${ROCM_PATH:-/opt/rocm}
gate=${BUILD_GATE:-build-gate.sh}
turn=${GPU_TURN:-gpu-turn.sh}
holder=${TRELLIS_HOLDER:-trellis-ref-glq}
vectors=${GLQ_VECTORS:-glq_vectors.bin}
mkdir -p .glq-build
export TMPDIR="$PWD/.glq-build"
export BUILD_GATE_NAME="$holder" BUILD_MAX_HOLD=90

output=".glq-build/test_glq_$target"
case "$target" in
    cpu)
        "$gate" build "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
            glq.cpp test_glq.cpp -o "$output"
        "$output" "$vectors"
        ;;
    sm_120|sm_89)
        "$gate" build "${NVCC:-nvcc}" -std=c++17 -O2 -arch="$target" \
            -DGLQ_TEST_GPU -x cu glq.cpp glq_gpu.cu test_glq.cpp -o "$output"
        if [[ ${RUN_GPU:-0} == 1 ]]; then "$turn" "$holder" 45 "$output" "$vectors"; fi
        ;;
    gfx1036|gfx1151)
        "$gate" build "${CLANGXX:-clang++}" --offload-arch="$target" --rocm-path="$sdk" \
            --hip-device-lib-path="$sdk/amdgcn/bitcode" -std=c++17 -O2 -Werror \
            -DGLQ_TEST_GPU -x hip glq.cpp glq_gpu.cu test_glq.cpp \
            -L"$sdk/lib" -Wl,-rpath,"$sdk/lib" \
            -Wl,-rpath-link,"$sdk/lib${HIP_RUNTIME_DEPS:+:$HIP_RUNTIME_DEPS}" -lamdhip64 -o "$output"
        if [[ ${RUN_GPU:-0} == 1 ]]; then
            "$turn" "$holder" 45 env LD_LIBRARY_PATH="$sdk/lib${HIP_RUNTIME_DEPS:+:$HIP_RUNTIME_DEPS}" \
                "$output" "$vectors"
        fi
        ;;
esac
