#!/bin/sh
set -eu
export LC_ALL=C
cli=${1:?usage: test_checkpoint.sh CHECKPOINT_CLI [cpu|cuda|rocm]}
backend=${2:-cpu}
case "$backend" in cpu|cuda|rocm) ;; *) exit 2 ;; esac
work=$(mktemp -d "${TMPDIR:-/arboretum}/exl3-checkpoint-test.XXXXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

fail() {
    if "$@" 2>"$work/error"; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
}

zeros() { dd if=/dev/zero bs="$1" count=1 status=none; }
header() {
    h=$1
    while [ "$((${#h} % 8))" != 0 ]; do h="$h "; done
    printf '%b' "$(printf '\\%03o\\%03o' "$((${#h} % 256))" "$((${#h} / 256))")"
    zeros 6
    printf '%s' "$h"
}

# Encode original-basis weights with distinct rows and columns. A one-hot
# activation at input 17 must select row 17 up to quantization error.
awk 'BEGIN {
    # F32 LE for -2, -1, 0, 1, 2.
    word[0] = "\000\000\000\300"; word[1] = "\000\000\200\277"
    word[2] = "\000\000\000\000"; word[3] = "\000\000\200\077"
    word[4] = "\000\000\000\100"
    for (r = 0; r < 128; ++r)
        for (c = 0; c < 128; ++c)
            printf "%s", word[(r * 13 + c * 7 + r * c * 11) % 5]
}' >"$work/weights"
{ zeros 68; printf '\000\000\200\077'; zeros 440; } >"$work/x"
for book in 3inst mcg; do
    "$cli" encode "$book" 16 model.proj "$work/weights" "$work/$book.safetensors"
    "$cli" forward "$backend" "$work/$book.safetensors" model.proj "$work/x" "$work/$book.y"
    od -An -v --endian=little -t f4 "$work/$book.y" | awk '{
        for (i = 1; i <= NF; ++i) {
            expected = (17 * 13 + (7 + 17 * 11) * c++) % 5 - 2
            delta = $i - expected
            error += delta * delta
            energy += expected * expected
        }
    }
    END {
        if (c != 128 || error >= 0.04 * energy) {
            printf "encode/forward: output length %d, relative L2 error %.4f\n", c, sqrt(error / energy) > "/dev/stderr"
            exit 1
        }
    }'
done

# Exact zero matrix and fractional-rate MUL1 marker.
zeros 65536 >"$work/zero"
zeros 512 >"$work/zero.y"
"$cli" encode mul1 3 layer "$work/zero" "$work/zero.safetensors"
"$cli" forward "$backend" "$work/zero.safetensors" layer "$work/x" "$work/y"
cmp "$work/zero.y" "$work/y"

# Independent 256x256 native fixture. The four 128x128 rotated blocks have
# constant states 0, 65535, 65535, 0; pins MUL1 1.5-bit values -3.453125 and
# -0.6513671875 from the v1.6.7 reference. Packed signs flip input 0 and
# output 128. With x[0]=1, x[128]=2 and bias=1, the expected non-bias
# contributions are 128*(-a+2*b) and -128*(-b+2*a) at outputs 0 and 128.
fixture='{"layer.su":{"dtype":"I16","shape":[16],"data_offsets":[0,32]},"layer.sv":{"dtype":"I16","shape":[16],"data_offsets":[32,64]},"layer.bias":{"dtype":"F16","shape":[256],"data_offsets":[64,576]},"layer.mul1":{"dtype":"I32","shape":[],"data_offsets":[576,580]},"layer.trellis":{"dtype":"I16","shape":[16,16,24],"data_offsets":[580,12868]}}'
zeros 384 >"$work/low"
tr '\000' '\377' <"$work/low" >"$work/high"
{
    printf '\000\074'
    i=1
    while [ "$i" -lt 256 ]; do
        printf '\000\074'
        i=$((i+1))
    done
} >"$work/bias"
{
    printf '\001'; zeros 31      # packed su
    zeros 16; printf '\001'; zeros 15  # packed sv
    cat "$work/bias"
    printf '\055\321\334\203'     # I32 multiplier
    i=0
    while [ "$i" -lt 8 ]; do
        cat "$work/low" "$work/high"
        i=$((i+1))
    done
    i=0
    while [ "$i" -lt 8 ]; do
        cat "$work/high" "$work/low"
        i=$((i+1))
    done
} >"$work/payload"
{ header "$fixture"; cat "$work/payload"; } >"$work/multi.safetensors"
{ printf '\000\000\200\077'; zeros 508; printf '\000\000\000\100'; zeros 508; } >"$work/multi.x"
{
    printf '\000\040\212\103' # 276.25
    i=1
    while [ "$i" -lt 256 ]; do
        if [ "$i" -eq 128 ]; then printf '\000\150\110\104'; # 801.625
        else printf '\000\000\200\077'; fi
        i=$((i+1))
    done
} >"$work/multi.expected"
"$cli" forward "$backend" "$work/multi.safetensors" layer "$work/multi.x" "$work/multi.y"
cmp "$work/multi.expected" "$work/multi.y"

printf 'preserve' >"$work/existing"
fail "$cli" encode 3inst 16 model.proj "$work/weights" "$work/existing"
fail "$cli" forward cpu "$work/multi.safetensors" layer "$work/multi.x" "$work/existing"
[ "$(cat "$work/existing")" = preserve ]
ln -s "$work/missing" "$work/symlink"
fail "$cli" encode 3inst 16 model.proj "$work/weights" "$work/symlink"
[ -L "$work/symlink" ] && [ ! -e "$work/missing" ]

cp "$work/multi.safetensors" "$work/bad-marker"
printf '\000' | dd of="$work/bad-marker" bs=1 seek="$((8 + (${#fixture}+7)/8*8 + 576))" conv=notrunc status=none
{ header "$(printf '%s' "$fixture" | sed 's/\[580,12868\]/[579,12867]/')"; cat "$work/payload"; } >"$work/overlap"
{ header "$(printf '%s' "$fixture" | sed 's/"dtype":"I16"/"dtype":"I16","dtype":"I16"/')"; cat "$work/payload"; } >"$work/duplicate"
head -c -1 "$work/multi.safetensors" >"$work/truncated"
for bad in bad-marker overlap duplicate truncated; do
    fail "$cli" forward cpu "$work/$bad" layer "$work/multi.x" "$work/rejected"
    [ ! -e "$work/rejected" ]
done
fail "$cli" forward cpu "$work/multi.safetensors" layer "$work/x" "$work/rejected"
if [ "$backend" != cuda ]; then
    fail "$cli" forward cuda "$work/multi.safetensors" layer "$work/multi.x" "$work/rejected"
    grep -q 'CUDA backend not built' "$work/error"
fi
if [ "$backend" != rocm ]; then
    fail "$cli" forward rocm "$work/multi.safetensors" layer "$work/multi.x" "$work/rejected"
    grep -q 'ROCm backend not built' "$work/error"
fi
[ ! -e "$work/rejected" ]
