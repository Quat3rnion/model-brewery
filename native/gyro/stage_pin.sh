#!/bin/sh
set -eu

# Agention Gyro 587a5269848eb07996c9c1433f9ad638604eb4d8, Apache-2.0.
src=$1
dest=$2
mkdir -p "$dest"
work=$(mktemp -d "$dest/stage.XXXXXXXX")
trap 'rm -rf "$work"' EXIT
for f in tq_ref.h tq_ref.c tq_codebook.c; do cp "$src/src/$f" "$work/$f"; done
(
    cd "$work"
    sha256sum --check --status <<'EOF'
81a2f2b0760379794dd67aebf2db453dbf94b99a65232d8ec36080180e20333c  tq_ref.h
2a2c4d0bc6f7d7a4653482c791c3cc99b5a066ad90e85e7e69d0ff78f6e9ba1c  tq_ref.c
0eb4beccdf0f03e9a6ea6912a50d7ac63d5ea537675646d32a196eb10b2d1c13  tq_codebook.c
EOF
) || { echo "Agention upstream pin mismatch" >&2; exit 1; }
cp "$src/LICENSE" "$src/NOTICE" "$work/"
{
    head -n 3 "$work/tq_codebook.c"
    sed -n '/^const uint16_t tq_codebook_f16/,/^};/{ /^    /p; }' "$work/tq_codebook.c"
} > "$work/gyro_codebook.inc"
for f in tq_ref.h tq_ref.c tq_codebook.c gyro_codebook.inc LICENSE NOTICE; do
    if ! cmp -s "$work/$f" "$dest/$f"; then mv -f "$work/$f" "$dest/$f"; fi
done
