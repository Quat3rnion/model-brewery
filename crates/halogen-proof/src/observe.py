"""Run inside the one qualification container, never against a supplied URL."""
import json
import os
import socket
import sys
import time


def proc_fields(pid):
    # comm may contain whitespace or parentheses.
    with open(f"/proc/{pid}/stat", encoding="ascii") as f:
        return f.read().rsplit(")", 1)[1].split()


def owned_engine(executable, ancestor, checkpoint, port):
    candidates = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            if os.readlink(f"/proc/{name}/exe") != executable:
                continue
            with open(f"/proc/{name}/cmdline", "rb") as f:
                args = f.read().split(b"\0")
            for key, value in (("--ck", checkpoint), ("--port", port)):
                flag = key.encode()
                if args.count(flag) != 1 or args[args.index(flag) + 1] != value.encode():
                    break
            else:
                fields = proc_fields(name)
                parent = int(fields[1])
                while parent > 0 and parent != ancestor:
                    parent = int(proc_fields(parent)[1])
                if parent == ancestor:
                    candidates.append((int(name), fields[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"expected one owned flash_serve, found {len(candidates)}")
    return candidates[0]


def listener(pid, port):
    address = f"0100007F:{int(port):04X}"
    sockets = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            sockets.add(os.readlink(f"/proc/{pid}/fd/{fd}"))
        except FileNotFoundError:
            continue
    with open(f"/proc/{pid}/net/tcp", encoding="ascii") as f:
        next(f)
        inodes = [cols[9] for cols in map(str.split, f) if cols[1] == address and cols[3] == "0A"]
    if len(inodes) != 1 or f"socket:[{inodes[0]}]" not in sockets:
        raise ValueError("IPv4 loopback listener does not uniquely belong to flash_serve")
    return inodes[0]


def mapped_files(pid, expected):
    seen = {entry["path"]: [] for entry in expected}
    by_identity = {(e["major"], e["minor"], e["inode"]): e["path"] for e in expected}
    with open(f"/proc/{pid}/maps", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            addr, perms, offset, device, inode, *name = line.rstrip("\n").split(maxsplit=5)
            path = name[0] if name else ""
            major, minor = (int(x, 16) for x in device.split(":"))
            identity = (major, minor, int(inode))
            bound_path = by_identity.get(identity)
            if bound_path is None and path not in seen:
                # Reject an extra HGN even when it comes from the image itself.
                if path.removesuffix(" (deleted)").lower().endswith(".hgn"):
                    raise ValueError(f"undeclared HGN mapping: {path}")
                continue
            if path != bound_path or not perms.startswith("r--"):
                raise ValueError(f"wrong inode, path, or permissions for mapping: {path}")
            start, end = (int(x, 16) for x in addr.split("-"))
            offset = int(offset, 16)
            seen[path].append((offset, offset + end - start))
    for e in expected:
        covered = 0
        for start, end in sorted(seen[e["path"]]):
            if start > covered:
                break
            covered = max(covered, end)
        if covered < e["bytes"]:
            raise ValueError(f"{e['path']} is not fully mapped read-only")


def pong(port):
    with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5) as connection:
        connection.sendall(b"PING\n")
        # Fixed total deadline and length, even if the peer drips bytes.
        deadline = time.monotonic() + 0.5
        reply = bytearray()
        while len(reply) < 5:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Halogen PONG timed out")
            connection.settimeout(remaining)
            data = connection.recv(5 - len(reply))
            if not data:
                break
            reply += data
        if reply != b"PONG\n":
            raise ValueError("expected Halogen PONG")


def main(executable, ancestor, port, seconds, expected_json):
    expected = json.loads(expected_json)
    deadline = time.monotonic() + int(seconds)
    last = "engine never observed"
    while time.monotonic() < deadline:
        try:
            engine = owned_engine(executable, int(ancestor), expected[0]["path"], port)
            pid = engine[0]
            inode = listener(pid, port)
            mapped_files(pid, expected)
            pong(port)
            if engine != owned_engine(executable, int(ancestor), expected[0]["path"], port):
                raise ValueError("flash_serve changed during PING")
            if listener(pid, port) != inode:
                raise ValueError("listener changed during PING")
            mapped_files(pid, expected)
            print(json.dumps({"engine_pid": pid, "start_ticks": int(engine[1]),
                              "listener_inode": int(inode), "mapped": expected}))
            return
        except (OSError, ValueError) as e:
            last = str(e)
            time.sleep(0.15)
    raise SystemExit(f"load identity not established: {last}")


if __name__ == "__main__":
    main(*sys.argv[1:])
