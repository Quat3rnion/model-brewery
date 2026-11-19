#!/usr/bin/sh
# Installed twice, with the release payload digest substituted by the packager.
set -eu
case "${0##*/}" in
    brewery-weight-edit) helper=weight_edit ;;
    brewery-weight-tune) helper=weight_tune ;;
    *) exit 64 ;;
esac
assets=${0%/*}/brewery-weight-runtime
probe=0
packed=0
if [ "$#" = 1 ] && [ "$1" = --capabilities ]; then
    probe=1
    work=/
else
    work=$(/usr/bin/pwd -P)
    case "${work##*/}" in
        .weight-gguf-?*) names='source.gguf' ;;
        .weight-hgn-?*) names='source.hgn' ;;
        .weight-safetensors-?*) names='source.safetensors source.hf' ;;
        *) exit 64 ;;
    esac
    test "$(/usr/bin/stat -c '%a:%u' -- "$work")" = "700:$(/usr/bin/id -u)" || exit 64
    if [ "$helper" = weight_edit ]; then
        case "${1:-}" in inspect|inventory|tile|propose|apply) ;; *) exit 64 ;; esac
        source=${2:-}
    else
        source=${1:-}
    fi
    allowed=0
    for name in $names; do
        if [ "$source" = "$name" ] || [ "$source" = "$work/$name" ]; then
            test ! -L "$work/$name" && test -e "$work/$name" || exit 64
            allowed=1
        fi
    done
    test "$allowed" = 1 || exit 64
    case "${BREWERY_WEIGHT_PACKED:-0}" in 0) ;; 1) packed=1 ;; *) exit 64 ;; esac
fi
# Only the exact scratch directory (if any) is mounted writable. No host HOME,
# model root, controller state, credentials, network, or inherited Python path.
python=/usr/bin/python3
bootstrap='
import hashlib, json, os, pathlib, sys
helper, probe, pin, packed, *args = sys.argv[1:]
root = pathlib.Path("/helper")
receipt = (root / "manifest.json").read_bytes()
if hashlib.sha256(receipt).hexdigest() != pin:
    sys.exit("weight runtime manifest pin does not match")
manifest = json.loads(receipt)
if manifest["abi"] != sys.implementation.cache_tag:
    sys.exit("weight runtime Python ABI changed; rebuild the release")
files = manifest["files"]
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        sys.exit("symlink in weight runtime")
    if not path.is_dir():
        actual.add(str(path.relative_to(root)))
if actual != set(files) | {"manifest.json"}:
    sys.exit("weight runtime file set does not match")
for name, digest in files.items():
    path = root / name
    if not path.is_file() or path.is_symlink():
        sys.exit("nonregular file in weight runtime")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            sys.exit("weight runtime file does not match manifest: " + name)
sys.path.insert(0, str(root))
os.environ["BREWERY_WEIGHT_PACKED"] = packed
from importlib import import_module
module = import_module(helper)
if probe == "1":
    jobs = ["weight_tune"] if helper == "weight_tune" else ["weight_inspect", "weight_propose", "weight_apply"]
    print(json.dumps(dict(schema="brewery.weight-runtime.v1", jobs=jobs,
                          formats=["gguf", "hgn", "safetensors", "hf"], visual=True, packed=True)))
else:
    sys.argv = [module.__file__, *args]
    sys.exit(module.main())
'
# Add the optional work bind without reinterpreting the engine arguments.
if [ "$probe" = 0 ]; then
    set -- --bind "$work" "$work" --chdir "$work" --setenv HOME "$work" --setenv TMPDIR "$work" \
        -- "$python" -ISB -c "$bootstrap" "$helper" "$probe" '@MANIFEST_SHA256@' "$packed" "$@"
else
    set -- --chdir / -- "$python" -ISB -c "$bootstrap" "$helper" "$probe" '@MANIFEST_SHA256@' "$packed"
fi
exec /usr/bin/env -i /usr/bin/bwrap --unshare-all --unshare-user --disable-userns --die-with-parent \
    --clearenv --cap-drop ALL --ro-bind /usr/lib /usr/lib --ro-bind-try /usr/lib64 /usr/lib64 \
    --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 --ro-bind "$python" "$python" \
    --dev /dev --ro-bind "$assets" /helper --setenv LC_ALL C --setenv OMP_NUM_THREADS 1 \
    --setenv MKL_NUM_THREADS 1 --setenv OPENBLAS_NUM_THREADS 1 "$@" </dev/null
