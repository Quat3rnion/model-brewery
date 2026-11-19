#!/usr/bin/env python3
"""Find the local Brewery controller and serve it through a private Unix socket.

Run as the same UID as Brewery and Caddy, in the same network namespace.
The controller must explicitly set BREWERY_DATA and an IPv4 BREWERY_LISTEN.
"""

import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import sys
import tempfile


RUN = Path("/mnt/games/model-brewery/local-sm120")
PROC = Path("/proc")
UID = os.getuid()
LOOPBACK = "127.0.0.1"
EXE = RUN / "bin/brewery"


def identify(proc, exes, data):
    """Return the configured port (zero means ephemeral), or None."""
    try:
        if proc.stat().st_uid != UID:
            return None
        link = proc / "exe"
        exe = Path(os.readlink(link))
        if exe not in exes or exe.resolve(strict=True) != exe:
            return None
        installed = exe.stat()
        if (installed.st_uid != UID or not stat.S_ISREG(installed.st_mode)
                or installed.st_mode & 0o022 or not link.samefile(exe)
                or not (proc / "ns/net").samefile(PROC / "self/ns/net")):
            return None
        fields = (proc / "environ").read_bytes().split(b"\0")
        if [f for f in fields if f.startswith(b"BREWERY_DATA=")] != [
            b"BREWERY_DATA=" + os.fsencode(data)
        ]:
            return None
        listen = [f for f in fields if f.startswith(b"BREWERY_LISTEN=")]
        if len(listen) == 1:
            match = re.fullmatch(rb"BREWERY_LISTEN=127\.0\.0\.1:([0-9]{1,5})", listen[0])
            if match and int(match[1]) <= 65535:
                return int(match[1])
    except (OSError, RuntimeError):
        pass
    return None


def socket_inodes(proc):
    with os.scandir(proc / "fd") as fds:
        for fd in fds:
            try:
                link = os.readlink(fd.path)
            except FileNotFoundError:
                continue
            if link.startswith("socket:[") and link.endswith("]"):
                yield link[8:-1]


def tcp4():
    with (PROC / "self/net/tcp").open() as table:
        next(table)
        for row in table:
            fields = row.split()
            local, remote, state, inode = fields[1], fields[2], fields[3], fields[9]
            if inode != "0":
                yield local, remote, state, inode


def proc_address(addr):
    ip, port = addr
    return f"{int.from_bytes(socket.inet_aton(ip), sys.byteorder):08X}:{port:04X}"


def discover(exes, data):
    targets = []
    for proc in PROC.iterdir():
        if not proc.name.isdecimal():
            continue
        configured = identify(proc, exes, data)
        if configured is not None:
            try:
                owned = set(socket_inodes(proc))
            except OSError:
                continue
            targets.append((proc, configured, owned))
    if len(targets) != 1:
        return None
    proc, configured, owned = targets[0]
    address = proc_address((LOOPBACK, 0)).split(":")[0]
    ports = [int(local.split(":")[1], 16) for local, _, state, inode in tcp4()
             if state == "0A" and inode in owned and local.startswith(address + ":")]
    if len(ports) == 1 and configured in (0, ports[0]):
        return proc, ports[0]
    return None


def verified_peer(sock, proc, exes, data):
    """Check the accepted fd, not merely the process that owned the listener."""
    local = proc_address(sock.getsockname())
    remote = proc_address(sock.getpeername())
    client_inode = str(os.fstat(sock.fileno()).st_ino)
    rows = {(src, dst, inode) for src, dst, state, inode in tcp4()
            if state == "01" and (src, dst) in ((local, remote), (remote, local))}
    if (local, remote, client_inode) not in rows:
        return False
    peers = {inode for src, dst, inode in rows if (src, dst) == (remote, local)}
    return len(peers) == 1 and not peers.isdisjoint(socket_inodes(proc)) and (
        identify(proc, exes, data) is not None
    )


async def dial(target, exes, data):
    proc, port = target
    pinned = os.pidfd_open(int(proc.name))
    try:
        configured = identify(proc, exes, data)
        if configured not in (0, port):
            raise ConnectionError("Brewery process changed")
        with socket.socket() as sock:
            sock.setblocking(False)
            loop = asyncio.get_running_loop()
            async with asyncio.timeout(1):
                await loop.sock_connect(sock, (LOOPBACK, port))
                # Linux exposes the peer inode only after the server accepts.
                while not select.select([pinned], [], [], 0)[0]:
                    if verified_peer(sock, proc, exes, data):
                        if not select.select([pinned], [], [], 0)[0]:
                            # Transfer an already-verified TCP connection, with no data
                            # ever sent to the ephemeral address before verification.
                            return await asyncio.open_connection(sock=sock.dup())
                    await asyncio.sleep(0.01)
    finally:
        os.close(pinned)
    raise ConnectionError("Brewery process exited")


async def copy(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    writer.write_eof()


def publish_port(path, port):
    # Caddy reads the entire file for each request. Rename prevents partial reads.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".port-", delete=False) as file:
        pending = Path(file.name)
        try:
            file.write(str(port).encode("ascii"))
            file.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


async def serve(path, exes, data):
    target = None
    active = 0
    stopping = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stopping.set)

    async def bridge(reader, writer):
        nonlocal active
        if target is None or active >= 64:
            writer.close()
            return
        active += 1
        upstream = None
        tasks = []
        try:
            incoming, upstream = await dial(target, exes, data)
            tasks = [asyncio.create_task(copy(reader, upstream)),
                     asyncio.create_task(copy(incoming, writer))]
            await asyncio.gather(*tasks)
        except (OSError, TimeoutError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            if upstream is not None:
                upstream.close()
            writer.close()
            active -= 1

    port_file = path.with_name(path.name + ".port")
    publish_port(port_file, 0)
    try:
        async with await asyncio.start_unix_server(bridge, path, backlog=64):
            port = 0
            while not stopping.is_set():
                target = discover(exes, data)
                current = target[1] if target else 0
                if current != port:
                    publish_port(port_file, current)
                    port = current
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=0.5)
                except TimeoutError:
                    pass
    finally:
        path.unlink(missing_ok=True)
        publish_port(port_file, 0)


def serve_locked(path, exes, data):
    if path.resolve() != path or not private_dir(path.parent):
        raise ValueError("the socket must be in a canonical 0700 directory owned by the current UID")
    os.umask(0o077)
    # Only replace a leftover socket from our own instance, never a live listener.
    lock = path.with_name(path.name + ".lock")
    with os.fdopen(os.open(lock, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)) as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            meta = path.lstat()
            if meta.st_uid != UID or not stat.S_ISSOCK(meta.st_mode):
                raise ValueError("refusing to replace non-socket path")
            # Only a cooperating shim is expected in this private directory.
            path.unlink()
        asyncio.run(serve(path, exes, data))


def private_dir(path):
    meta = path.stat()
    return (path.resolve(strict=True) == path and meta.st_uid == UID
            and stat.S_ISDIR(meta.st_mode) and stat.S_IMODE(meta.st_mode) == 0o700)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "status"])
    parser.add_argument("--exe", type=Path, action="append", default=[],
                        help="Also trust this exact canonical controller executable (repeatable).")
    parser.add_argument("--data-dir", type=Path, default=RUN / "controller")
    parser.add_argument("--socket", type=Path, help="Private Unix socket path for the Caddy upstream.")
    args = parser.parse_args()
    if any(not exe.is_absolute() for exe in args.exe):
        parser.error("--exe must be an absolute canonical path")
    if (args.command == "serve") != (args.socket is not None):
        parser.error("--socket is required only with serve")
    exes = (EXE, *args.exe)
    try:
        if not private_dir(args.data_dir):
            parser.error("--data-dir must be a canonical 0700 directory owned by the current UID")
        if args.command == "serve":
            serve_locked(args.socket, exes, args.data_dir)
            return 0
        target = discover(exes, args.data_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if target is None:
        print("No unique verified loopback Brewery listener.", file=sys.stderr)
        return 1
    print(f"{LOOPBACK}:{target[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
