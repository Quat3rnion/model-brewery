"""Stop the private container if its job stops renewing the lease."""
import os
import signal
import sys
import time


def fresh(path, window):
    try:
        with open(path, encoding="ascii") as lease:
            return 0 <= time.monotonic_ns() - int(lease.read(32)) < window * 10**9
    except (OSError, ValueError):
        return False


def watch(owner, path, window, grace):
    while os.getppid() == owner and fresh(path, window):
        time.sleep(1)
    if os.getppid() != owner:
        return
    os.kill(owner, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while os.getppid() == owner and time.monotonic() < deadline:
        time.sleep(1)
    if os.getppid() == owner:
        os.kill(owner, signal.SIGKILL)


if __name__ == "__main__":
    path = "/brewery-lease/current"
    if not fresh(path, 20):
        raise SystemExit("worker lease missing or expired")
    owner = os.getpid()
    if os.fork() == 0:
        watch(owner, path, 20, 30)
        os._exit(0)
    os.execv("/usr/local/bin/entrypoint.sh", ["/usr/local/bin/entrypoint.sh", sys.argv[1]])
