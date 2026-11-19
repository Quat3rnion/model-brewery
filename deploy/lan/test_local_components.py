import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).with_name("local_components.py")
spec = importlib.util.spec_from_file_location("local_components", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class Admission(unittest.TestCase):
    def test_four_recipe_envelopes_and_combined_roles(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            source = root / "base"
            components = root / "components"
            state = root / "state"
            output = state / "jobs/verified-id/work/converted"
            source.mkdir()
            components.mkdir()
            output.mkdir(parents=True)
            vision = components / "vision.safetensors"
            vision.touch()
            name = "model.layers.0.fixture.weight"
            body = {"schema": 1, "name": "fixture é", "architecture": "qwen4_exp",
                    "catalog_sha256": "a" * 64, "base": "v2", "overrides": {name: "ht43"}}
            revision = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()
            def inp(path, kind):
                return {"host": "local-components", "path": str(path), "kind": kind}
            catalog = {"catalog_sha256": "a" * 64, "architecture": "qwen4_exp",
                       "tensors": [{"name": name, "editable": True}]}
            with patch.multiple(ops, COMPONENTS=components, STATE=state, HOME=root), \
                    patch.object(ops, "source_path", return_value=source), \
                    patch.object(ops, "read_catalog", return_value=catalog, create=True), \
                    patch.object(ops, "ram_available", return_value=20 << 30), \
                    patch.object(ops, "disk_available", return_value=300_000_000_000):
                for kind in ("inspect", "plan", "convert", "verify"):
                    job = {"kind": kind, "inputs": [inp(source, "hf")],
                           "options": {"profile_recipe": {**body, "revision": revision}}}
                    if kind == "verify":
                        job["inputs"].append(inp(output, "hgn"))
                        job["options"]["decode_backend"] = "cpu"
                    else:
                        job["options"]["profile"] = "v2"
                    req = {"op": "prepare", "action": "start", "job": job}
                    with self.subTest(kind=kind):
                        ops.precheck(req)
                        job["options"]["vision"] = len(job["inputs"])
                        job["inputs"].append(inp(vision, "vision"))
                        ops.precheck(req)
                        job["options"]["recipe_path"] = str(vision)
                        with self.assertRaisesRegex(ValueError, "option"):
                            ops.precheck(req)
                        del job["options"]["recipe_path"]
                        job["options"]["profile_recipe"] = {**body, "revision": "b" * 64}
                        with self.assertRaisesRegex(ValueError, "revision"):
                            ops.precheck(req)
                job["options"]["profile_recipe"] = {**body, "revision": revision}
                job["inputs"][1] = inp(components, "hgn")
                with self.assertRaisesRegex(ValueError, "conversion artifact"):
                    ops.precheck(req)

    def test_prepare_fails_before_input_resolution_for_cross_host_or_low_ram(self):
        with tempfile.TemporaryDirectory(dir="/arboretum") as directory:
            root = Path(directory)
            source = root / "base"
            source.mkdir()
            components = root / "components"
            components.mkdir()
            vision = components / "vision.safetensors"
            vision.touch()
            job = {"kind": "convert", "inputs": [
                {"host": "local-components", "kind": "hf", "path": str(source)},
                {"host": "local-components", "kind": "vision", "path": str(vision)},
            ], "options": {"vision": 1}}
            req = {"op": "prepare", "action": "start", "job": job}
            with patch.multiple(ops, COMPONENTS=components, HOME=root), \
                    patch.object(ops, "source_path", return_value=source):
                with patch.object(ops, "ram_available", return_value=1):
                    with self.assertRaisesRegex(ValueError, "16 GiB"):
                        ops.precheck(req)
                with patch.object(ops, "ram_available", return_value=20 << 30), \
                        patch.object(ops, "disk_available", return_value=300_000_000_000):
                    ops.precheck(req)
                    job["inputs"][0]["host"] = "local-sm120"
                    with self.assertRaisesRegex(ValueError, "Library alias"):
                        ops.precheck(req)
                    job["inputs"][0]["host"] = "local-components"
                    job["options"]["ht_backend"] = "cuda"
                    with self.assertRaisesRegex(ValueError, "CPU-only"):
                        ops.precheck(req)
                    del job["options"]["ht_backend"]
                    job["inputs"][1]["path"] = str(source)
                    with self.assertRaisesRegex(ValueError, "private components"):
                        ops.precheck(req)

    def test_config_preserves_original_worker_and_adds_only_component_host(self):
        original = {"hosts": {"local-sm120": {"config": "pinned", "capabilities": {"convert": True}}}}
        source = "/test/pinned-hf-snapshot"
        with patch.object(ops, "read_pin", return_value={"source": source}):
            config, dispatch = ops.configs(original)
        self.assertEqual(dispatch["hosts"]["local-sm120"], original["hosts"]["local-sm120"])
        self.assertEqual(set(dispatch["hosts"]), {"local-sm120", "local-components"})
        self.assertEqual(config["pinned_publication"]["source"], source)
        self.assertEqual(config["source_roots"], [source, str(ops.COMPONENTS)])
        self.assertEqual(config["engine_env"]["BREWERY_COMPONENT_CPU_ONLY"], "1")
        self.assertNotEqual(config["engine_env"]["BREWERY_CUSTOM_CONVERTER"],
                            config["engine_env"]["BREWERY_CONVERTER"])
        self.assertNotIn("BREWERY_EXPECTED_SOURCE_SHA256", config["engine_env"])
        allowed = {k for k, v in dispatch["hosts"]["local-components"]["capabilities"].items() if v}
        self.assertEqual(allowed, {"inspect", "plan", "convert", "verify", "convert_components", "custom_profiles"})


if __name__ == "__main__":
    unittest.main()
