#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ $# -lt 2 || $# -gt 3 || ( $1 != stock && $1 != rocmfpx ) || ! ${3:-0} =~ ^[0-9]+$ ]]; then
    echo 'usage: run.sh {stock|rocmfpx} {cpu|cuda|hip} [REGISTRY_DEVICE_INDEX=0]' >&2
    exit 2
fi
family=$1 backend=$2 index=${3:-0}
case $backend in
    cpu) registry=CPU ;;
    cuda) registry=CUDA ;;
    hip) registry=ROCm ;;
    *) echo "invalid backend: $backend" >&2; exit 2 ;;
esac

# The process is only the native probe. Shell/jq drive the canonical catalog.
catalog=../../crates/controller/src/gguf_catalog.json
jq -e '
    .schema == 1 and (.tensor_types |
        length == 32 and (map(.id) | unique | length == 32) and
        (map(.type_id) | unique | length == 32) and
        all(.[]; (.id | test("^[A-Z0-9_]+$")) and
            (.type_id | type == "number") and (.block_size > 0) and (.block_bytes > 0) and
            (.requires_imatrix | type == "boolean") and
            ((.encoder_family // "stock") | IN("stock", "rocmfpx"))))
' "$catalog" >/dev/null
rows=$(jq -er --arg family "$family" '
    .tensor_types[] | select(
        if $family == "stock" then (.encoder_family // "stock") == "stock"
        else .rocmfpx_supported != false end) |
    [.id, .type_id, .block_size, .block_bytes, (if .requires_imatrix then 1 else 0 end)] | @tsv
' "$catalog")
revision_key=encoder_revision
expected_count=27
if [[ $family == rocmfpx ]]; then
    revision_key=rocmfpx_encoder_revision
    expected_count=31
fi
[[ $(wc -l <<< "$rows") == "$expected_count" ]]
expected_pin=$(jq -er --arg key "$revision_key" '.[$key]' "$catalog")
bin=".build/$family-$backend"
missing=''
if [[ ! -x $bin || ! -r $bin.revision ]]; then
    missing=missing_backend_binary
else
    IFS=$'\t' read -r pin embedded < "$bin.revision"
    actual=$("$bin" --revision) || actual=unreadable
    if [[ $pin != "$expected_pin" || $actual != "$embedded" ]] ||
       [[ $actual != unknown && ( ! $actual =~ ^[0-9a-f]{7,40}$ || $pin != "$actual"* ) ]]; then
        missing=backend_revision_mismatch
    fi
fi

ulimit -c 0
printf 'family\tbackend\tformat\ttype_id\tstatus\tdevice\toutputs\tmax_abs_error\tmax_tolerance_ratio\treason\n'
failed=0
while IFS=$'\t' read -r format id block bytes imatrix; do
    if [[ -n $missing ]]; then
        result=$'FAIL\t-\t-\t-\t-\t'"$missing"
        failed=1
    elif result=$(timeout 25s "$bin" "$registry" "$index" "$format" "$id" "$block" "$bytes" "$imatrix"); then
        if [[ $result != PASS$'\t'* || $result == *$'\n'* ]]; then
            result=$'FAIL\t-\t-\t-\t-\tinvalid_probe_output'
            failed=1
        fi
    else
        code=$?
        [[ $result == FAIL$'\t'* && $result != *$'\n'* ]] ||
            result=$'FAIL\t-\t-\t-\t-\tprobe_exit_'"$code"
        failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$family" "$backend" "$format" "$id" "$result"
done <<< "$rows"
exit "$failed"
