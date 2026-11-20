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
proof_fds=0
pinned=0
if [ "$#" = 1 ] && [ "$1" = --capabilities ]; then
    probe=1
    work=/
    if /usr/bin/bwrap --help | /usr/bin/grep -q -- ' --ro-bind-fd '; then
        proof_fds=1
    fi
else
    work=$(/usr/bin/pwd -P)
    case "${work##*/}" in
        .weight-gguf-?*) names='source.gguf' ;;
        .weight-hgn-?*) names='source.hgn' ;;
        .weight-safetensors-?*) names='source.safetensors source.hf' ;;
        *) exit 64 ;;
    esac
    test "$(/usr/bin/stat -c '%a:%u' -- "$work")" = "700:$(/usr/bin/id -u)" || exit 64
    if [ "${1:-}" = --pinned-safetensors ]; then
        case "$work" in /weight-source|/weight-source/*) exit 64 ;; esac
        case "${work##*/}" in .weight-safetensors-?*) pinned=1 ;; *) exit 64 ;; esac
    elif [ "$helper" = weight_edit ]; then
        case "${1:-}" in inspect|inventory|tile|propose|apply) ;; *) exit 64 ;; esac
        source=${2:-}
    else
        source=${1:-}
    fi
    if [ "$pinned" = 0 ]; then
        allowed=0
        for name in $names; do
            if [ "$source" = "$name" ] || [ "$source" = "$work/$name" ]; then
                test ! -L "$work/$name" && test -e "$work/$name" || exit 64
                allowed=1
            fi
        done
        test "$allowed" = 1 || exit 64
    fi
    case "${BREWERY_WEIGHT_PACKED:-0}" in 0) ;; 1) packed=1 ;; *) exit 64 ;; esac
fi
# Only the exact scratch directory (if any) is mounted writable. No host HOME,
# model root, controller state, credentials, network, or inherited Python path.
python=/usr/bin/python3
bootstrap='
import hashlib, json, os, pathlib, sys
helper, probe, pin, packed, proof_fds, *args = sys.argv[1:]
@VERIFY_RUNTIME@
verify_runtime(pathlib.Path("/helper"), pin)
os.environ["BREWERY_WEIGHT_PACKED"] = packed
from importlib import import_module
module = import_module(helper)
if probe == "1":
    jobs = ["weight_tune"] if helper == "weight_tune" else ["weight_inspect", "weight_propose", "weight_apply"]
    print(json.dumps(dict(schema="brewery.weight-runtime.v1", jobs=jobs,
                           formats=["gguf", "hgn", "safetensors", "hf"], visual=True, packed=True,
                           proof_source_fds=proof_fds == "1")))
else:
    sys.argv = [module.__file__, *args]
    sys.exit(module.main())
'
# Add the optional work bind without reinterpreting the engine arguments.
if [ "$probe" = 0 ]; then
    set -- --bind "$work" "$work" --chdir "$work" --setenv HOME "$work" --setenv TMPDIR "$work" \
        -- "$python" -ISB -c "$bootstrap" "$helper" "$probe" '@MANIFEST_SHA256@' "$packed" "$proof_fds" "$@"
else
    set -- --chdir / -- "$python" -ISB -c "$bootstrap" "$helper" "$probe" '@MANIFEST_SHA256@' "$packed" "$proof_fds"
fi
set -- /usr/bin/bwrap --unshare-all --unshare-user --disable-userns --die-with-parent \
    --clearenv --cap-drop ALL --ro-bind /usr/lib /usr/lib --ro-bind-try /usr/lib64 /usr/lib64 \
    --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 --ro-bind "$python" "$python" \
    --dev /dev --ro-bind "$assets" /helper --setenv LC_ALL C --setenv OMP_NUM_THREADS 1 \
    --setenv MKL_NUM_THREADS 1 --setenv OPENBLAS_NUM_THREADS 1 "$@"
if [ "$pinned" = 1 ]; then
    # Use only inherited member descriptors, never resolve a host source path.
    exec /usr/bin/env -i "$python" -ISB -c '
import fcntl, json, os, re, stat, sys

def require(ok):
    if not ok:
        raise ValueError("invalid pinned SafeTensors sandbox source")

def prepare(helper, work, *argv):
    marker = argv.index("--pinned-safetensors")
    require(re.fullmatch(r"[1-9][0-9]{0,3}", argv[marker + 1]))
    count = int(argv[marker + 1])
    end = marker + 2 + 2 * count
    require(count <= 4096 and argv[end] == "--")
    fields, args = argv[marker + 2:end], argv[end + 1:]
    root = "/weight-source"
    source = args[1 if helper == "weight_edit" else 0]
    hf = source == root + "/source.hf"
    require(hf or (source == root + "/source.safetensors" and count == 1))
    if helper == "weight_edit":
        require(args[0] in ("inspect", "inventory", "tile", "propose", "apply"))
    manifest = work + "/source-manifest.json"
    require(args[-2:] == ("--source-manifest", manifest))
    with os.fdopen(os.open(manifest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as file:
        before = os.fstat(file.fileno())
        limit = 64 << 20
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_uid == os.geteuid() and before.st_mode & 0o077 == 0
                and 0 < before.st_size <= limit)
        raw = file.read(limit + 1)
        after = os.fstat(file.fileno())
        require(len(raw) == before.st_size and before.st_ctime_ns == after.st_ctime_ns)
    rows = json.loads(raw)
    require(isinstance(rows, list) and len(rows) == count)
    fds, names = set(), set()
    mounts = ["--tmpfs", root]
    if hf:
        mounts.extend(("--dir", source))
    for fd_text, name, row in zip(fields[::2], fields[1::2], rows, strict=True):
        require(re.fullmatch(r"[1-9][0-9]{0,9}", fd_text)
                and name not in ("", ".", "..") and name.isprintable()
                and "/" not in name and "\\" not in name and len(os.fsencode(name)) <= 255)
        fd = int(fd_text)
        require(3 <= fd <= 2147483647 and fd not in fds and name not in names
                and row["path"] == name)
        fds.add(fd)
        names.add(name)
        fd_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        info = os.fstat(fd)
        require(fd_flags & (os.O_ACCMODE | os.O_PATH) == os.O_RDONLY
                and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_size == row["bytes"])
        if "device" in row or "inode" in row:
            require((info.st_dev, info.st_ino) == (row["device"], row["inode"]))
        else:
            require("chunk_sha256" in row)
        require(info.st_ino != before.st_ino or info.st_dev != before.st_dev)
        mounts.extend(("--ro-bind-fd", fd_text, source + "/" + name if hf else source))
    mounts.extend(("--remount-ro", root))
    split = argv.index("--")
    os.execv(argv[0], (*argv[:split], *mounts, *argv[split:marker], *args))

try:
    prepare(*sys.argv[1:])
except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
    sys.exit("weight runtime: " + str(exc))
' "$helper" "$work" "$@" </dev/null
fi
exec /usr/bin/env -i "$@" </dev/null
