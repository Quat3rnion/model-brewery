import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Bootstrap(unittest.TestCase):
    def test_private_no_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as root:
            target = Path(root) / "admin"
            command = [sys.executable, "-B", str(Path(__file__).with_name("bootstrap_admin.py")), str(target)]
            result = subprocess.run(command, capture_output=True, check=True)
            token = (target / "operator.key").read_bytes().strip()
            self.assertRegex(token, rb"^[a-f0-9]{64}$")
            self.assertNotIn(token, result.stdout + result.stderr)
            verify = target / "verify.sha256"
            self.assertEqual(verify.read_text().strip(), hashlib.sha256(token).hexdigest())
            for path, mode in ((target, 0o700), (verify, 0o600), (target / "operator.key", 0o600)):
                self.assertEqual(path.stat().st_mode & 0o777, mode)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((target / "operator.key").read_bytes().strip(), token)
