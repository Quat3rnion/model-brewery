#!/usr/bin/env bash
# Capture bounded encoder activity from NVIDIA's stock CUPTI injection sample.
set -euo pipefail
if (( $# < 4 )) || [[ $3 != -- || $2 != /* || ! -f $2 || $4 != /* || ! -x $4 ]]; then
    echo "Usage: bash $0 NEW_EVIDENCE_DIR /ABS/CUPTI_COLLECTOR.so -- /ABS/CONVERTER [ARGS...]" >&2
    exit 2
fi

out=$1
collector=$2
shift 3
umask 077
mkdir -- "$out"
export LC_ALL=C
{
    printf 'started_utc=%s\nhost=%s\ncwd=%q\n' "$(date -uIs)" "$(hostname)" "$PWD"
    printf 'CUDA_VISIBLE_DEVICES=%q\nCUDA_DISABLE_PTX_JIT=%q\nCUDA_FORCE_PTX_JIT=%q\n' \
        "${CUDA_VISIBLE_DEVICES-<unset>}" "${CUDA_DISABLE_PTX_JIT-<unset>}" "${CUDA_FORCE_PTX_JIT-<unset>}"
    printf 'command='; printf '%q ' "$@"; printf '\n'
    sha256sum -- "$collector" "$1"
} > "$out/run.txt"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.total,memory.free \
    --format=csv > "$out/devices.csv" 2> "$out/device-query.stderr" || true
printf 'encoder_activity=not-observed\nrun_finished=no\n' > "$out/status.txt"
touch -- "$out/encoder-samples.txt" "$out/diagnostics.txt"

# Keep the converter running to completion even after reaching the sample cap.
# Runtime/API calls, launches without a measured duration, and decode are not
# encoder evidence. The following deviceId line is part of the stock record.
set +e
env CUDA_INJECTION64_PATH="$collector" "$@" 2>&1 |
    TRACE_OUT="$out" awk '
        function save(line) { print substr(line,1,4096) >> sample }
        BEGIN {
            root=ENVIRON["TRACE_OUT"]
            sample=root "/encoder-samples.txt"
            diag=root "/diagnostics.txt"
            counts=root "/encoder-counts.tsv"
        }
        /^CONCURRENT_KERNEL \[/ {
            kind=""
            if ($0 !~ /duration [1-9][0-9]*, "/) next
            if ($0 ~ /encode_tilesILi3E/) kind="ht3"
            else if ($0 ~ /encode_tilesILi4E/) kind="ht4"
            else if ($0 ~ /i4r_quantize/) kind="i4r"
            else if ($0 ~ /q4_quantize/) kind="q4cp"
            record=$0
            next
        }
        kind != "" && /^[[:space:]]+grid \[/ { next }
        kind != "" && /^[[:space:]]+deviceId [0-9]+,.*channelType COMPUTE/ {
            if (++seen[kind] <= 32) { save(record); save($0) }
        }
        { kind="" }
        /^[[:space:]]+(deviceId|sourceGraphId|executionModel|grid) / ||
            /^(DRIVER|RUNTIME|MEMCPY|MEMSET|OVERHEAD|MEMORY2)[[:space:]]/ { next }
        kept++ < 128 { print substr($0,1,4096) >> diag }
        END {
            print "codec\tcompleted_activity_records" > counts
            printf "ht3\t%d\nht4\t%d\ni4r\t%d\nq4cp\t%d\n",seen["ht3"],seen["ht4"],seen["i4r"],seen["q4cp"] > counts
        }
    '
status=("${PIPESTATUS[@]}")
set -e
observed=not-observed
if (( status[1] == 0 )) && awk 'NR>1 && $2>0 {found=1} END {exit !found}' "$out/encoder-counts.tsv"; then
    observed=observed
fi
{
    printf 'encoder_activity=%s\nrun_finished=yes\n' "$observed"
    printf 'command_exit=%s\ncollector_parse_exit=%s\nfinished_utc=%s\n' "${status[@]}" "$(date -uIs)"
} > "$out/status.new"
mv -- "$out/status.new" "$out/status.txt"
(( status[0] == 0 )) || exit "${status[0]}"
(( status[1] == 0 )) || exit "${status[1]}"
[[ $observed == observed ]] || exit 3
