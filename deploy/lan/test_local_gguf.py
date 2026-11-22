from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import local_gguf as ops
import stage_local_gguf as stages


class Admission(unittest.TestCase):
    def test_stock_and_weight_prepare_ignore_unavailable_optional_paths(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            imports, state = root / "imports", root / "state"
            imports.mkdir()
            state.mkdir()
            source = imports / "model.gguf"
            source.write_bytes(b"GGUF")
            policy = {"imports": str(imports), "state": str(state)}
            local, standalone = root / "local-roots.json", root / "standalone.json"
            local.write_text(json.dumps(policy))
            standalone.write_text(json.dumps(policy | {
                "runtime": str(root / "removed-runtime"), "llama": str(root / "removed-llama"),
            }))
            for p in (local, standalone):
                p.chmod(0o600)
            work = state / "jobs/test/work"
            work.mkdir(parents=True)
            with patch.multiple(ops, LOCAL_ROOTS=local, STANDALONE=standalone), \
                    patch.object(ops, "headroom") as gate, \
                    patch.object(ops, "pinned_native"), \
                    patch.object(ops.sys, "argv", ["local_gguf.py", "--wait", "--", str(ops.ENGINE)]), \
                    patch.object(ops.Path, "cwd", return_value=work), \
                    patch.object(ops.os, "pread") as stdin, patch.object(ops.os, "execve") as launch, \
                    patch.object(ops, "catalog", return_value={"presets": [{"id": "Q4_0"}], "tensor_types": []}):
                for kind, options in (("convert", {"output_format": "gguf", "gguf_type": "Q4_0"}),
                                      ("weight_inspect", {"mode": "inventory", "offset": 0, "limit": 4})):
                    job = {"kind": kind, "options": options,
                           "inputs": [{"host": "local-gguf", "kind": "gguf", "path": str(source)}]}
                    ops.precheck({"op": "prepare", "action": "start", "job": job})
                    self.assertEqual(gate.call_args.args[-1], state)
                    stdin.return_value = json.dumps(job).encode()
                    ops.admit()
                    _, argv, env = launch.call_args.args
                    self.assertFalse(any("removed-runtime" in arg or "removed-llama" in arg for arg in argv))
                    self.assertTrue({"BREWERY_GGUF_HF_EXPORTER", "BREWERY_GGUF_PYTHON", "BREWERY_LLAMA_CPP"}
                                    .isdisjoint(env))

    def test_standalone_execution_refuses_changed_runtime_before_engine(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            state, imports = root / "state", root / "imports"
            work = state / "jobs/test/work"
            work.mkdir(parents=True)
            imports.mkdir()
            source, assets = imports / "model.safetensors", imports / "assets"
            source.write_bytes(b"payload")
            assets.mkdir()
            (assets / "config.json").write_text("{}")
            runtime, llama = root / "runtime", root / "llama"
            (runtime / "bin").mkdir(parents=True)
            llama.mkdir()
            # Emulate a runtime whose version preflight exits nonzero.
            python = runtime / "bin/python"
            python.write_text("#!/bin/sh\nexit 1\n")
            python.chmod(0o700)
            job = {"kind": "convert", "options": {"output_format": "gguf", "gguf_type": "BF16"},
                   "inputs": [{"host": "local-gguf", "kind": kind, "path": str(path)}
                              for kind, path in (("safetensors", source), ("assets", assets))]}
            standalone = root / "standalone.json"
            standalone.write_text(json.dumps({
                        "imports": str(imports), "state": str(state), "runtime": str(runtime), "llama": str(llama),
            }))
            standalone.chmod(0o600)
            with patch.multiple(ops, IMPORTS=imports, STATE=state, STANDALONE=standalone), \
                    patch.object(ops, "headroom"), patch.object(ops, "pinned_native"), \
                    patch.object(ops, "catalog", return_value={"presets": [{"id": "BF16"}], "tensor_types": []}), \
                    patch.object(ops.sys, "argv", ["local_gguf.py", "--wait", "--", str(ops.ENGINE)]), \
                    patch.object(ops.os, "pread", return_value=json.dumps(job).encode()), \
                    patch.object(ops.Path, "cwd", return_value=work), \
                    patch.object(ops, "sandbox", side_effect=lambda work, paths, cmd, *a, **k: (cmd, {})), \
                    patch.object(ops.os, "execve") as launch:
                with self.assertRaisesRegex(ValueError, "runtime"):
                    ops.admit()
                launch.assert_not_called()

    def test_standalone_accepts_only_explicit_direct_assets(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            source = root / "source.safetensors"
            source.write_bytes(b"weights")
            assets = root / "metadata"
            assets.mkdir()
            (assets / "config.json").write_text("{}")
            (assets / "tokenizer.json").write_text("{}")
            job = {"kind": "convert", "options": {"output_format": "gguf", "gguf_type": "Q4_0"},
                   "inputs": [{"host": "local-gguf", "kind": kind, "path": str(path)}
                              for kind, path in (("safetensors", source), ("assets", assets))]}
            catalog = {"presets": [{"id": "Q4_0"}], "tensor_types": []}
            with patch.object(ops, "IMPORTS", root), patch.object(ops, "catalog", return_value=catalog):
                with self.assertRaisesRegex(ValueError, "not staged"):
                    ops.job_inputs(job, standalone=False)
                self.assertEqual(ops.job_inputs(job, standalone=True), [source, assets])
                (assets / "foreign.py").write_text("no")
                with self.assertRaisesRegex(ValueError, "metadata"):
                    ops.job_inputs(job, standalone=True)
                (assets / "foreign.py").unlink()
                (assets / "tokenizer.json").unlink()
                (assets / "tokenizer.json").symlink_to(source)
                with self.assertRaisesRegex(ValueError, "regular files"):
                    ops.job_inputs(job, standalone=True)

    def test_failed_runtime_probe_never_proposes_or_edits_dispatch(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            release, runtime, llama = (root / n for n in ("release", "runtime", "llama"))
            for path in (release, runtime / "bin", llama):
                path.mkdir(parents=True)
            for leaf in ("brewery-worker", "brewery-engine"):
                (release / leaf).touch()
            (runtime / "bin/python").touch()
            quantizer = root / "quantizer"
            quantizer.touch()
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"encoder_revision": stages.REVISION}))
            dispatch = root / "dispatch.json"
            dispatch.write_text('{"hosts":{"other":{"local":true}}}')
            original = dispatch.read_bytes()
            with patch.object(stages, "QUANT_SHA", stages.file_sha(quantizer)), \
                    patch.object(stages, "probe_export", side_effect=ValueError("failed probe")):
                with self.assertRaisesRegex(ValueError, "failed probe"):
                    stages.stage(root / "staged", release, quantizer, catalog, dispatch,
                                 runtime=runtime, llama=llama, probe_source=root / "model.safetensors",
                                 probe_assets=root / "assets")
            self.assertFalse((root / "staged").exists())
            self.assertEqual(dispatch.read_bytes(), original)

    def test_only_a_pinned_fork_that_passes_convert_and_verify_is_proposed(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            release = root / "release"
            release.mkdir()
            for leaf in ("brewery-worker", "brewery-engine"):
                (release / leaf).touch()
            stock, fork, probe, catalog, dispatch = (root / n for n in (
                "quantizer", "fork", "tiny.gguf", "catalog.json", "dispatch.json"))
            stock.write_bytes(b"stock")
            fork.write_bytes(b"fork")
            fork.chmod(0o700)
            probe.write_bytes(b"tiny")
            revision = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c"
            values = {"encoder_revision": stages.REVISION, "rocmfpx_encoder_revision": revision}
            catalog.write_text(json.dumps(values))
            original = b'{"hosts":{}}'
            dispatch.write_bytes(original)
            staged = root / "staged"
            pin = stages.file_sha(fork)
            args = (staged, release, stock, catalog, dispatch)
            options = dict(rocmfpx_quantizer=fork, rocmfpx_sha256=pin, rocmfpx_probe_source=probe)
            with patch.multiple(stages, QUANT_SHA=stages.file_sha(stock), ROCMFPX_QUANT_SHA=pin), \
                    patch.object(stages, "probe_rocmfpx", side_effect=ValueError("verify failed")) as smoke:
                with self.assertRaisesRegex(ValueError, "together"):
                    stages.stage(*args, rocmfpx_quantizer=fork)
                with self.assertRaisesRegex(ValueError, "approved CPU build"):
                    stages.stage(*args, **(options | {"rocmfpx_sha256": "f" * 64}))
                catalog.write_text(json.dumps(values | {"rocmfpx_encoder_revision": "0" * 40}))
                with self.assertRaisesRegex(ValueError, "revision"):
                    stages.stage(*args, **options)
                catalog.write_text(json.dumps(values))
                smoke.assert_not_called()
                with self.assertRaisesRegex(ValueError, "verify failed"):
                    stages.stage(*args, **options)
                self.assertFalse(staged.exists())
                smoke.side_effect = None
                stages.stage(*args, **options)
            self.assertEqual(dispatch.read_bytes(), original)
            self.assertEqual((staged / "bin/llama-quantize").read_bytes(), b"stock")
            fork_staged = staged / "bin/llama-quantize-rocmfpx"
            self.assertEqual(fork_staged.read_bytes(), b"fork")
            self.assertEqual((staged / "rocmfpx.sha256").read_text().strip(), pin)
            caps = json.loads((staged / "dispatch.proposed.json").read_text())["hosts"]["local-gguf"]["capabilities"]
            self.assertIs(caps["gguf_output"], True)
            self.assertIs(caps["gguf_rocmfpx_output"], True)
            self.assertNotIn("gguf_safetensors_output", caps)
            cfg = json.loads((staged / "worker.json").read_text())
            self.assertEqual(cfg["engine_env"], {"BREWERY_GGUF_ROCMFPX_QUANTIZER": str(fork_staged),
                                               "BREWERY_GGUF_ROCMFPX_SHA256": pin})

    def test_upgrade_keeps_existing_roots_and_workers_but_drops_stale_fork(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            old = root / "old"
            release, runtime, llama = (root / n for n in ("release", "runtime", "llama"))
            for path in (release, runtime / "bin", llama, old / "bin", old / "imports", old / "worker"):
                path.mkdir(parents=True)
            for leaf in ("brewery-worker", "brewery-engine"):
                (release / leaf).touch()
            (runtime / "bin/python").touch()
            wrapper = old / "bin/local_gguf.py"
            wrapper.touch()
            worker = old / "worker.json"
            worker.write_text(json.dumps({
                "root": str(old / "worker"), "source_roots": [str(old / "imports")],
                "admission": str(wrapper), "disk_reserve_bytes": 20 << 30,
                "engine_env": {
                    "KEEP": "untouched",
                    "BREWERY_GGUF_ROCMFPX_QUANTIZER": str(old / "bin/llama-quantize-rocmfpx"),
                    "BREWERY_GGUF_ROCMFPX_SHA256": "f" * 64,
                },
            }))
            quantizer, catalog, dispatch = (root / n for n in ("quantizer", "catalog.json", "dispatch.json"))
            quantizer.touch()
            catalog.write_text(json.dumps({"encoder_revision": stages.REVISION}))
            hosts = {"another": {"ssh": "fixture-unchanged"}, "local-gguf": {
                "name": "Existing", "local": True, "worker": str(wrapper), "config": str(worker),
                "capabilities": {"gguf_output": True, "gguf_hf_output": False, "gguf_rocmfpx_output": True},
            }}
            dispatch.write_text(json.dumps({"hosts": hosts}))
            original = (worker.read_bytes(), dispatch.read_bytes())
            with patch.object(stages, "QUANT_SHA", stages.file_sha(quantizer)), \
                    patch.object(stages, "probe_export"):
                staged = root / "staged"
                stages.stage(staged, release, quantizer, catalog, dispatch,
                             runtime=runtime, llama=llama, probe_source=root / "fixture.safetensors",
                             probe_assets=root / "assets")
            self.assertEqual((worker.read_bytes(), dispatch.read_bytes()), original)
            proposed = json.loads((staged / "dispatch.proposed.json").read_text())["hosts"]
            self.assertEqual(proposed["another"], hosts["another"])
            self.assertEqual(proposed["local-gguf"]["name"], "Existing")
            self.assertEqual(proposed["local-gguf"]["capabilities"], {
                "gguf_output": True, "gguf_hf_output": False, "gguf_safetensors_output": True,
            })
            cfg = json.loads((staged / "worker.json").read_text())
            self.assertEqual(cfg["source_roots"], [str(old / "imports")])
            self.assertEqual(cfg["root"], str(old / "worker"))
            self.assertEqual(cfg["disk_reserve_bytes"], 20 << 30)
            self.assertEqual(json.loads((staged / "local-roots.json").read_text()), {
                "imports": str(old / "imports"), "state": str(old / "worker"),
            })
            self.assertEqual(cfg["engine_env"]["KEEP"], "untouched")
            self.assertNotIn("BREWERY_GGUF_ROCMFPX_QUANTIZER", cfg["engine_env"])
            self.assertNotIn("BREWERY_GGUF_ROCMFPX_SHA256", cfg["engine_env"])

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

    def test_stock_only_upgrade_preserves_admission_roots_without_standalone(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            old, release = root / "old", root / "release"
            for path in (release, old / "imports", old / "worker", old / "bin/weight-runtime-packed"):
                path.mkdir(parents=True)
            for leaf in ("brewery-worker", "brewery-engine"):
                (release / leaf).touch()
            wrapper = old / "bin/local_gguf.py"
            wrapper.touch()
            (old / "bin/weight-runtime-packed/edit").write_text("bound runtime")
            cfg = old / "worker.json"
            cfg.write_text(json.dumps({"root": str(old / "worker"), "admission": str(wrapper),
                                       "source_roots": [str(old / "imports")]}))
            stock, catalog, dispatch = (root / n for n in ("quantizer", "catalog.json", "dispatch.json"))
            stock.touch()
            catalog.write_text(json.dumps({"encoder_revision": stages.REVISION, "presets": [{"id": "Q4_0"}],
                                           "tensor_types": []}))
            caps = {"gguf_output": True, "gguf_safetensors_output": False, "weight_inspect": True}
            dispatch.write_text(json.dumps({"hosts": {"local-gguf": {
                "local": True, "worker": str(wrapper), "config": str(cfg), "capabilities": caps,
            }}}))
            staged = root / "staged"
            with patch.object(stages, "QUANT_SHA", stages.file_sha(stock)):
                stages.stage(staged, release, stock, catalog, dispatch)
            self.assertFalse((staged / "standalone.json").exists())
            self.assertEqual((staged / "bin/weight-runtime-packed/edit").read_text(), "bound runtime")
            proposed = json.loads((staged / "dispatch.proposed.json").read_text())["hosts"]["local-gguf"]
            self.assertEqual(proposed["capabilities"], caps)
            source = old / "imports/fixture.gguf"
            source.write_bytes(b"GGUF")
            job = {"kind": "convert", "options": {"output_format": "gguf", "gguf_type": "Q4_0"},
                   "inputs": [{"host": "local-gguf", "kind": "gguf", "path": str(source)}]}
            with patch.multiple(ops, ROOT=staged, LOCAL_ROOTS=staged / "local-roots.json"), \
                    patch.object(ops, "headroom", side_effect=ValueError("RAM gate reached")):
                with self.assertRaisesRegex(ValueError, "RAM gate reached"):
                    ops.precheck({"op": "prepare", "action": "start", "job": job})

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

    def test_fork_options_are_separate_and_need_the_staged_binary(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as name:
            root = Path(name)
            inp, fork, pin = (root / n for n in ("source.gguf", "llama-quantize-rocmfpx", "rocmfpx.sha256"))
            inp.write_bytes(b"GGUF")
            catalog = {
                "rocmfpx_encoder_revision": "fb08d7cdb67082ddda15a25e64dc7ec413afc41c",
                "presets": [{"id": "Q8_0"}, {"id": "Q2_0", "rocmfpx_supported": False}],
                "tensor_types": [{"id": "Q3_0_ROCMFPX", "encoder_family": "rocmfpx"}],
            }
            options = {"output_format": "gguf", "gguf_type": "Q8_0",
                       "tensor_types": {"blk.0.attn_q.weight": "Q3_0_ROCMFPX"}}
            job = {"kind": "convert", "inputs": [{"host": "local-gguf", "kind": "gguf", "path": str(inp)}],
                   "options": options}
            req = {"op": "prepare", "action": "start", "job": job}
            with patch.multiple(ops, IMPORTS=root, ROCMFPX_PIN=pin, ROCMFPX_QUANTIZER=fork), \
                    patch.object(ops, "catalog", return_value=catalog), patch.object(ops, "headroom"):
                with self.assertRaisesRegex(ValueError, "encoder family"):
                    ops.precheck(req)
                options.update(encoder_family="rocmfpx", encoder_revision="0" * 40)
                with self.assertRaisesRegex(ValueError, "revision"):
                    ops.precheck(req)
                options["encoder_revision"] = catalog["rocmfpx_encoder_revision"]
                with self.assertRaisesRegex(ValueError, "not staged"):
                    ops.precheck(req)
                fork.write_bytes(b"fork")
                fork.chmod(0o700)
                pin.write_text(stages.file_sha(fork) + "\n")
                pin.chmod(0o600)
                ops.precheck(req)
                options["gguf_type"] = "Q2_0"
                with self.assertRaisesRegex(ValueError, "not supported by ROCmFPX"):
                    ops.precheck(req)
                options["gguf_type"] = "Q8_0"
                fork.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "approved CPU build"):
                    ops.precheck(req)
                del options["encoder_family"], options["encoder_revision"], options["tensor_types"]
                ops.precheck(req)
