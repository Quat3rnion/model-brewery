#!/usr/bin/python3
"""Create a separate one-time controller administrator key without printing it."""
import hashlib
import os
from pathlib import Path
import secrets
import sys


def bootstrap(path):
    path = path.absolute()
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError("key directory parent must be canonical")
    os.umask(0o077)
    path.mkdir(mode=0o700)
    token = secrets.token_hex(32).encode("ascii")
    try:
        for name, value in (("operator.key", token), ("verify.sha256", hashlib.sha256(token).hexdigest().encode("ascii"))):
            with (path / name).open("xb") as out:
                out.write(value + b"\n")
                out.flush()
                os.fsync(out.fileno())
        for directory in (path, path.parent):
            fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        for name in ("operator.key", "verify.sha256"):
            (path / name).unlink(missing_ok=True)
        path.rmdir()
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bootstrap_admin.py NEW_PRIVATE_DIRECTORY")
    try:
        bootstrap(Path(sys.argv[1]))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
