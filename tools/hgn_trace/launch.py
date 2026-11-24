"""Isolated entry point for the worker's hash-locked CPU environment."""
from importlib.metadata import version
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
if sys.version_info[:3] != (3, 12, 13) or sys.prefix == sys.base_prefix:
    raise SystemExit("HGN trace needs the pinned CPython 3.12.13 virtual environment")
for line in (root / "requirements.lock.txt").read_text().splitlines():
    if line and not line[0].isspace() and not line.startswith("#"):
        name, expected = line.split()[0].split("==")
        if version(name) != expected:
            raise SystemExit(f"HGN trace package version mismatch: {name}")
sys.path.insert(0, str(root))
entry = "trace.py"
if "--stored-checks" in sys.argv:
    sys.argv.remove("--stored-checks")
    entry = "checks.py"
runpy.run_path(str(root / entry), run_name="__main__")
