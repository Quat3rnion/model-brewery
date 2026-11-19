#!/usr/bin/env bash
# Public wrapper regression. Replays the collector format; needs no GPU.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(mktemp -d "${TMPDIR:-/arboretum}/conversion-trace-test.XXXXXXXX")
trap 'rm -rf -- "$root"' EXIT
wrapper=$here/conversion-kernel-trace.sh
collector=$root/placeholder.so
touch "$collector"

record() {
    printf 'CONCURRENT_KERNEL [ 1, 2 ] duration %s, "%s", correlationId 7\n' "$1" "$2"
    printf '\tgrid [ 1, 1, 1 ], block [ 1, 1, 1 ]\n'
    printf '\tdeviceId 0, contextId 1, channelType COMPUTE, priority 0\n'
}
{
    printf 'RUNTIME [ 1, 2 ] duration 1, "cudaLaunchKernel", i4r_quantize\n'
    record 0 i4r_quantize
    record 1 decode_row_axis
} > "$root/no-encode"
status=0
bash "$wrapper" "$root/negative" "$collector" -- /usr/bin/cat "$root/no-encode" || status=$?
[[ $status == 3 ]]
grep -Fxq 'encoder_activity=not-observed' "$root/negative/status.txt"
[[ ! -s "$root/negative/encoder-samples.txt" ]]

for _ in {1..40}; do record 1 _Z12encode_tilesILi4ELb1EE; done > "$root/encode"
bash "$wrapper" "$root/positive" "$collector" -- /usr/bin/cat "$root/encode"
grep -Fxq 'encoder_activity=observed' "$root/positive/status.txt"
grep -Fxq $'ht4\t40' "$root/positive/encoder-counts.tsv"
[[ $(grep -c '^CONCURRENT_KERNEL' "$root/positive/encoder-samples.txt") == 32 ]]

status=0
bash "$wrapper" "$root/failed" "$collector" -- /usr/bin/bash -c 'cat "$1"; exit 2' _ "$root/encode" || status=$?
[[ $status == 2 ]]
grep -Fxq 'command_exit=2' "$root/failed/status.txt"
grep -Fxq 'encoder_activity=observed' "$root/failed/status.txt"
echo "conversion trace wrapper regression passed"
