"""Linux integration tests for the LAN upstream; no real controller changes."""

import os
import json
import select
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest


SHIM = Path(__file__).with_name("supervise.py")
FIXTURE = """
import socket
import sys

with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(sys.argv[1])))
    listener.listen()
    if len(sys.argv) > 2:
        import subprocess
        listener.set_inheritable(True)
        child = subprocess.Popen([sys.argv[2], "-c", '''
import socket
import sys

with socket.socket(fileno=int(sys.argv[1])) as listener:
    with listener.accept()[0] as client:
        print(client.recv(65536).hex(), flush=True)
''', str(listener.fileno())], pass_fds=(listener.fileno(),))
        print(listener.getsockname()[1], flush=True)
        child.wait()
        sys.exit()
    print(listener.getsockname()[1], flush=True)
    while True:
        client, _ = listener.accept()
        with client:
            while payload := client.recv(65536):
                client.sendall(payload)
"""


class TestSupervise(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = self.root / "data"
        self.data.mkdir(mode=0o700)
        self.exe = self.root / "controller"
        shutil.copyfile(sys.executable, self.exe)
        self.exe.chmod(0o700)
        self.args = ["--exe", str(self.exe), "--data-dir", str(self.data)]

    def fixture(self, exe=None, data=None, port=0, accept_elsewhere=False):
        args = [str(exe or self.exe), "-c", FIXTURE, str(port)]
        if accept_elsewhere:
            args.append(sys.executable)
        proc = self.enterContext(subprocess.Popen(
            args,
            env={**os.environ, "BREWERY_DATA": str(data or self.data),
                 "BREWERY_LISTEN": "127.0.0.1:0"},
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        ))
        self.addCleanup(self.stop, proc)
        return proc, int(proc.stdout.readline())

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)

    def status(self):
        return subprocess.run([sys.executable, "-B", SHIM, "status", *self.args],
                              capture_output=True, text=True, timeout=5)

    def serve(self):
        self.path = self.root / "upstream.sock"
        proc = self.enterContext(subprocess.Popen(
            [sys.executable, "-B", SHIM, "serve", *self.args, "--socket", str(self.path)],
            stderr=subprocess.PIPE, text=True,
        ))
        self.addCleanup(self.stop, proc)
        return proc

    def wait_port(self, expected, shim):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if shim.poll() is not None:
                self.fail(shim.stderr.read())
            port_file = self.path.with_suffix(".sock.port")
            if self.path.exists() and port_file.exists():
                if port_file.read_text() == str(expected):
                    return
            time.sleep(0.02)
        self.fail(f"shim did not publish {expected}")

    def dial(self):
        client = self.enterContext(socket.socket(socket.AF_UNIX))
        client.settimeout(5)
        client.connect(str(self.path))
        return client

    def test_status_uses_live_listener_for_the_exact_exe_and_data_directory(self):
        self.fixture(exe=sys.executable)
        self.fixture(data=self.root / "other-data")
        self.assertEqual(self.status().returncode, 1)

        _, port = self.fixture()
        result = self.status()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"127.0.0.1:{port}")

        self.fixture()
        self.assertEqual(self.status().returncode, 1, "multiple matches must be rejected")

    def test_bridge_refuses_a_connection_accepted_outside_verified_process(self):
        fixture, port = self.fixture(accept_elsewhere=True)
        shim = self.serve()
        self.wait_port(port, shim)
        with self.dial() as client:
            client.sendall(b"must not leave the Unix socket")
            try:
                self.assertEqual(client.recv(1), b"")
            except ConnectionResetError:
                pass
        self.assertEqual(fixture.wait(timeout=5), 0)
        self.assertEqual(fixture.stdout.read(), "\n", "unverified peer received bytes")
        self.assertTrue(select.select([shim.stderr], [], [], 2)[0], "missing bridge failure record")
        line = shim.stderr.readline()
        self.assertNotIn("must not leave", line)
        event = json.loads(line)
        self.assertEqual(event["event"], "brewery_bridge_failure")
        self.assertEqual(event["phase"], "verify_peer")
        self.assertEqual(event["reason"], "timeout")
        self.assertEqual(event["peer_check"], "peer_fd_not_owned")
        self.assertEqual(event["upstream_port"], port)
        self.assertGreaterEqual(event["elapsed_ms"], 900)
        self.assertEqual(event["to_upstream_bytes"], 0)

    def test_bridge_streams_to_a_verified_peer_and_recovers_after_restart(self):
        first, port = self.fixture()
        shim = self.serve()
        self.wait_port(port, shim)
        with self.dial() as client:
            for payload in (b"one", b"two"):
                client.sendall(payload)
                self.assertEqual(client.recv(len(payload), socket.MSG_WAITALL), payload)

            self.stop(first)
            self.assertEqual(client.recv(1), b"")

            # Reuse the old port with a process that the shim must never select.
            self.fixture(exe=sys.executable, port=port)
            self.wait_port(0, shim)
            with self.dial() as unavailable:
                try:
                    unavailable.sendall(b"not to this port")
                    self.assertEqual(unavailable.recv(1), b"")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            _, replacement_port = self.fixture()
            self.assertNotEqual(port, replacement_port)
            self.wait_port(replacement_port, shim)
            with self.dial() as reconnected:
                reconnected.sendall(b"ready")
                self.assertEqual(reconnected.recv(5, socket.MSG_WAITALL), b"ready")


if __name__ == "__main__":
    unittest.main()
