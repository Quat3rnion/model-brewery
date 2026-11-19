#!/usr/bin/env python3
"""Run the offline brewery.cross-container.v1 CPU tool."""
import os
from pathlib import Path
import sys


def main():
    path = Path(os.environ.get("BREWERY_HGN_IN_GGUF_BIN", ""))
    if not path.is_absolute() or not path.is_file():
        sys.exit("Set BREWERY_HGN_IN_GGUF_BIN to the absolute path of the built CPU helper.")
    os.execv(path, [str(path), *sys.argv[1:]])


if __name__ == "__main__":
    main()
