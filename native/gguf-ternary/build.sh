#!/usr/bin/env bash
# Output: CPU verifier, CUDA/HIP verifier named by architecture, and reusable GPU object.
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${OUT:?set scratch output directory}" "${GGML_INCLUDE:?}" "${GGML_LIB:?pinned static archive directory}"
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp"
common=(-std=c++17 -O2 -I"$GGML_INCLUDE")
oracle=(-Wl,--start-group "$GGML_LIB"/libggml{,-base,-cpu}.a -Wl,--end-group -lgomp -ldl -pthread)
case "${1:-}" in
cpu)
    "${CXX:-c++}" "${common[@]}" test.cpp "${oracle[@]}" -o "$OUT/cpu";;
sm_120)
    nvcc -std=c++17 -O2 -arch=sm_120 -c gemv.cu -o "$OUT/sm_120.o"
    cuda=${CUDA_HOME:-/opt/cuda}
    "${CXX:-c++}" "${common[@]}" -DTEST_CUDA -I"$cuda/include" test.cpp "$OUT/sm_120.o" \
        -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart "${oracle[@]}" -o "$OUT/sm_120";;
gfx1036|gfx1151)
    : "${ROCM_PATH:?source the local SDK env.sh}"
    "$ROCM_PATH/bin/hipcc" -x hip --offload-arch="$1" -std=c++17 -O2 -c gemv.cu -o "$OUT/$1.o"
    "${CXX:-c++}" "${common[@]}" -DTEST_HIP -D__HIP_PLATFORM_AMD__ -I"$ROCM_PATH/include" \
        test.cpp "$OUT/$1.o" -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 \
        "${oracle[@]}" -o "$OUT/$1";;
*) echo 'usage: bash build.sh {cpu|sm_120|gfx1036|gfx1151}' >&2; exit 2;;
esac
