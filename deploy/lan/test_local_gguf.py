from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import local_gguf as ops


class Admission(unittest.TestCase):
    def test_packed_hgn_reaches_unchanged_memory_gate_only_for_local_file(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            imports = Path(name)
            source = imports / "native.hgn"
            source.write_bytes(b"HGN1")
            job = {"kind": "weight_inspect",
                   "inputs": [{"path": str(source), "kind": "hgn", "host": "local-gguf"}],
                   "options": {"mode": "inventory", "offset": 0, "limit": 4, "packed": True}}
            req = {"op": "prepare", "action": "start", "job": job}
            with patch.object(ops, "IMPORTS", imports), patch.object(ops, "ram_available", return_value=1):
                with self.assertRaisesRegex(ValueError, "16 GiB"):
                    ops.precheck(req)
                job["inputs"][0]["kind"] = "gguf"
                with self.assertRaisesRegex(ValueError, "source"):
                    ops.precheck(req)

    def test_visual_and_reviewed_apply_paths_stay_inside_worker(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            imports = root / "imports"
            imports.mkdir()
            work = root / "state/jobs/producer/work"
            work.mkdir(parents=True)
            source, proposal = work / "edited.gguf", work / "weight-proposal.json"
            source.write_bytes(b"GGUF")
            proposal.write_bytes(b"{}")
            external = root / "foreign.gguf"
            external.write_bytes(b"GGUF")
            def inp(path, kind):
                return {"path": str(path), "kind": kind, "host": "local-gguf"}
            with patch.multiple(ops, IMPORTS=imports, STATE=root / "state"), patch.object(ops, "headroom"):
                job = {"kind": "weight_inspect", "inputs": [inp(source, "gguf")],
                       "options": {"mode": "inventory", "offset": 0, "limit": 4}}
                req = {"op": "prepare", "action": "start", "job": job}
                ops.precheck(req)
                job.update(kind="weight_apply", options={"source_sha256": "1" * 64, "proposal_sha256": "2" * 64},
                           inputs=[inp(source, "gguf"), inp(proposal, "weight-edit-proposal")])
                ops.precheck(req)
                job["inputs"][0] = inp(external, "gguf")
                with self.assertRaisesRegex(ValueError, "source"):
                    ops.precheck(req)
                job["inputs"][0] = inp(source, "gguf")
                job["inputs"][1] = inp(external, "weight-edit-proposal")
                with self.assertRaisesRegex(ValueError, "proposal"):
                    ops.precheck(req)

    def test_only_local_allowlisted_inputs_and_bounded_options(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            source = root / "model.gguf"
            source.write_bytes(b"GGUF")
            matrix = root / "matrix.gguf"
            matrix.write_bytes(b"GGUF")
            state = root / "state"
            published = state / "jobs/fake-producer/work/converted.gguf"
            published.parent.mkdir(parents=True)
            published.write_bytes(b"GGUF")
            catalog = {"presets": [{"id": "Q4_K_M"}, {"id": "IQ1_S", "requires_imatrix": True}],
                       "tensor_types": [{"id": "Q8_0"}]}
            def inp(path, kind, host="local-gguf"):
                return {"path": str(path), "kind": kind, "host": host}

            job = {"kind": "convert", "inputs": [inp(source, "gguf")],
                   "options": {"output_format": "gguf", "gguf_type": "IQ1_S",
                               "tensor_types": {"blk.0.attn_q.weight": "Q8_0"}}}
            req = {"op": "prepare", "action": "start", "job": job}
            with patch.multiple(ops, IMPORTS=root, STATE=state), patch.object(ops, "catalog", return_value=catalog), \
                    patch.object(ops, "headroom"):
                with self.assertRaisesRegex(ValueError, "imatrix required"):
                    ops.precheck(req)
                job["options"]["imatrix"] = 1
                job["inputs"].append(inp(matrix, "imatrix"))
                ops.precheck(req)
                for kind in ("inspect", "plan"):
                    job["kind"] = kind
                    ops.precheck(req)
                job["kind"] = "convert"
                job["inputs"][0]["host"] = "local-components"
                with self.assertRaisesRegex(ValueError, "local-gguf"):
                    ops.precheck(req)
                job["inputs"][0]["host"] = "local-gguf"
                job["options"]["tensor_types"] = {".*": "Q8_0"}
                with self.assertRaisesRegex(ValueError, "tensor"):
                    ops.precheck(req)
                job["options"]["tensor_types"] = {}
                job["options"]["vision"] = 1
                with self.assertRaisesRegex(ValueError, "option"):
                    ops.precheck(req)
                del job["options"]["vision"]
                job["kind"] = "verify"
                job["inputs"].insert(1, inp(published, "gguf"))
                job["options"]["imatrix"] = 2
                ops.precheck(req)
                wrong_output = root / "wrong.gguf"
                wrong_output.write_bytes(b"GGUF")
                job["inputs"][1] = inp(wrong_output, "gguf")
                with self.assertRaisesRegex(ValueError, "published"):
                    ops.precheck(req)

    def test_disk_and_ram_gate_cannot_be_overridden(self):
        with patch.object(ops, "ram_available", return_value=10 << 30):
            with self.assertRaisesRegex(ValueError, "16 GiB"):
                ops.headroom("convert", 1024)
        with patch.object(ops, "ram_available", return_value=22 << 30), \
                patch.object(ops.shutil, "disk_usage") as disk:
            disk.return_value.free = (16 << 30) + 1
            with self.assertRaisesRegex(ValueError, "free disk"):
                ops.headroom("convert", 1024)
