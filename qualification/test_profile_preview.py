from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name("profile_preview.py")
NATIVE = Path(os.environ.get(
    "BREWERY_PREVIEW_TEST_PLANS", "/arboretum/brewery-header-plan.y4P3876P"))
HYPER = "layers.0.attn_hyper_connection.block_inject_weight.weight"
EXPERT = "layers.1.mlp.experts.down_proj.weight"
NATIVE_AVAILABLE = all((NATIVE / f"{p}.json").is_file() for p in ("v2", "ht43"))


def fixture_plan(profile):
    # A small layout in the native JSON dialect: 448-byte header/directory,
    # then a one-row 64- or 68-byte weight and an aligned 2-byte tail.
    is_v2 = profile == "v2"
    identity = f"qwen3.8-flash-next-{profile}"
    encoding, size, tail = ("Q6G64", 64, 512) if is_v2 else ("Q8G64", 68, 576)
    tensors = [
        {"name": name, "encoding": store, "shape": shape,
         "origin": {"Tensor": {"name": f"model.language_model.{name}"}}, "rotation": None}
        for name, store, shape in [(HYPER, encoding, [1, 64]), ("norm.weight", "Bf16", [1])]
    ]
    entries = [
        {k: t[k] for k in ("name", "encoding", "shape")}
        | {"offset": offset, "size": length, "checksum": 0}
        for t, offset, length in zip(tensors, (448, tail), (size, 2))
    ]
    return {
        "summary": {
            "spec_version": "1.1.0", "spec_commit": "79e75fd4ae7642478823525b0a3212951f584af5",
            "profile": profile, "policy": f"bf16-hgn-{profile}-rtn-nf4-signed-rms-trellis-v1",
            "tensor_payloads_read": False, "architecture": "qwen4_exp", "assets": [],
            "source_format": "hf", "source_tensors": 2, "source_shards": 1,
            "source_dtypes": {"Bf16": 2}, "canonical_source_payload_bytes": 130,
            "output_hgn_bytes": tail + 2,
            "outputs": [{"filename": "model.hgn", "identity": identity,
                         "tensors": 2, "bytes": tail + 2, "stores": {encoding: 1, "Bf16": 1}}],
        },
        "plan": {"architecture": "qwen4_exp", "assets": [], "files": [
            {"filename": "model.hgn", "identity": identity, "tensors": tensors},
        ]},
        "layouts": [{"identity": identity, "data_offset": 448,
                     "file_size": tail + 2, "entries": entries}],
        "options": {"profile": profile, "seed": 0},
    }


class PreviewCLI(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def run_recipe(self, base, overrides, plans, version=1):
        recipe = self.folder / "recipe.json"
        recipe.write_text(json.dumps({
            "schema_version": version, "base_profile": base, "overrides": overrides,
        }), encoding="utf-8")
        return subprocess.run([
            sys.executable, str(CLI), "--recipe", str(recipe),
            "--v2-plan", str(plans / "v2.json"), "--ht43-plan", str(plans / "ht43.json"),
        ], capture_output=True, text=True, timeout=10)

    def preview(self, base, overrides, plans):
        result = self.run_recipe(base, overrides, plans)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def write_fixture(self, mutate=lambda _: None):
        for profile in ("v2", "ht43"):
            report = fixture_plan(profile)
            if profile == "ht43":
                mutate(report)
            (self.folder / f"{profile}.json").write_text(json.dumps(report), encoding="utf-8")

    def reject(self, base, overrides, error, version=1):
        result = self.run_recipe(base, overrides, self.folder, version=version)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn(error, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_hgn_prediction_includes_alignment_not_just_payload_difference(self):
        self.write_fixture()
        report = self.preview("v2", [{"select": {"layers": [0]}}], self.folder)
        self.assertEqual(report["size"]["files"], [{
            "filename": "model.hgn", "base_hgn_bytes": 514, "predicted_hgn_bytes": 578,
            "delta_hgn_bytes": 64, "base_overhead_bytes": 448, "predicted_overhead_bytes": 508,
        }])
        change, = report["resolved_recipe"]["changes"]
        self.assertEqual(change["before"], {"encoding": "Q6G64", "shape": [1, 64], "payload_bytes": 64})
        self.assertEqual(change["after"], {"encoding": "Q8G64", "shape": [1, 64], "payload_bytes": 68})
        self.assertEqual((change["tensor"], change["layer"], change["role"], change["kind"]),
                         (HYPER, 0, "hyper_connection", "override"))

    def test_rejects_ambiguous_unmatched_and_unsupported_overrides(self):
        self.write_fixture()
        for overrides, error in [
            ([{"select": {"layers": [0, 9]}}], "no compatible alternative in layer 9"),
            ([{"select": {"layers": [True]}}], "invalid layer indices"),
            ([{"select": {"tensors": [HYPER, "norm.weight"]}}], "no compatible alternative for tensor"),
            ([{"select": {"tensors": [EXPERT + "[0]"]}}], "no compatible alternative for tensor"),
            ([{"select": {"role": "attention"}}], "supported roles"),
            ([{"select": {"role": "routed_expert"}}], "unmatched selector"),
            ([{"select": {"tensors": [HYPER]}, "encoding": "I4r"}], "only a select object"),
            ([{"select": {"layers": [0]}}, {"select": {"tensors": [HYPER]}}],
             "conflicting selectors"),
        ]:
            with self.subTest(overrides=overrides):
                self.reject("v2", overrides, error)
        self.reject("v2", [], "expected recipe schema_version 1", version=True)
        self.reject("custom", [], "base_profile must be v2 or ht43")

    def test_rejects_incompatible_source_plan_and_layout_even_with_no_overrides(self):
        for mutate, error in [
            (lambda p: p["summary"].update(source_shards=2), "incompatible source metadata"),
            (lambda p: p["options"].update(seed=10), "incompatible native plan options"),
            (lambda p: p["options"].update(profile="v2"), "expected native ht43 plan"),
            (lambda p: p["plan"]["files"][0]["tensors"][0]["origin"]["Tensor"].update(name="other"),
             "incompatible source tensor or shape"),
            (lambda p: p["layouts"][0]["entries"][0].update(size=67), "inconsistent native layout"),
            (lambda p: p["layouts"][0]["entries"][-1].update(offset=568), "inconsistent native layout"),
        ]:
            with self.subTest(error=error):
                self.write_fixture(mutate)
                self.reject("v2", [], error)

    def test_rejects_oversize_input_before_decoding_json(self):
        self.write_fixture()
        with (self.folder / "ht43.json").open("r+b") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        self.reject("v2", [], "JSON exceeds 8388608 bytes")
        self.write_fixture()
        self.reject("v2", ["x" * (256 * 1024)], "JSON exceeds 262144 bytes")

    @unittest.skipUnless(NATIVE_AVAILABLE, "native plans not available; set BREWERY_PREVIEW_TEST_PLANS")
    def test_unmodified_native_plans_match_exact_hgn_totals_without_attesting_weights(self):
        for profile, size in (("v2", 118785840932), ("ht43", 109716416292)):
            with self.subTest(profile=profile):
                report = self.preview(profile, [], NATIVE)
                self.assertEqual(report["resolved_recipe"]["base"]["identity"],
                                 f"qwen3.8-flash-next-{profile}")
                self.assertEqual(report["size"]["predicted_hgn_bytes"], size)
                self.assertEqual(report["size"]["delta_hgn_bytes"], 0)
                self.assertEqual(report["resolved_recipe"]["changes"], [])
                self.assertEqual(report["source_compatibility"]["source_identity"], "unverified")
                self.assertEqual(report["source_compatibility"]["weight_identity"], "unverified")
                self.assertIs(report["executable"], False)
                self.assertIs(report["release_eligible"], False)

    @unittest.skipUnless(NATIVE_AVAILABLE, "native plans not available; set BREWERY_PREVIEW_TEST_PLANS")
    def test_native_expert_swap_moves_required_rotation_companions_in_both_directions(self):
        # Three native rows total 432,544,000 bytes in v2 and 317,849,600 in HT43.
        for profile, delta, before, after in (
            ("v2", -114694400, "I4r", "Ht3"), ("ht43", 114694400, "Ht3", "I4r"),
        ):
            with self.subTest(profile=profile):
                report = self.preview(profile, [{"select": {"tensors": [EXPERT]}}], NATIVE)
                changes = report["resolved_recipe"]["changes"]
                self.assertEqual({row["tensor"] for row in changes}, {
                    EXPERT, EXPERT.replace(".weight", ".suh"), EXPERT.replace(".weight", ".svh"),
                })
                weight = next(row for row in changes if row["kind"] == "override")
                self.assertEqual((weight["before"]["encoding"], weight["after"]["encoding"]),
                                 (before, after))
                self.assertEqual(weight["before"]["shape"], [512, 2560, 640])
                self.assertEqual(weight["before"]["shape"], weight["after"]["shape"])
                self.assertEqual(report["size"]["delta_hgn_bytes"], delta)

    @unittest.skipUnless(NATIVE_AVAILABLE, "native plans not available; set BREWERY_PREVIEW_TEST_PLANS")
    def test_native_equivalent_selectors_have_same_digest_and_canonical_change_sequence(self):
        first = self.preview("v2", [{"select": {"layers": [1, 0], "role": "hyper_connection"}}],
                             NATIVE)
        second = self.preview("v2", [{"select": {"layers": [0], "role": "hyper_connection"}},
                                     {"select": {"layers": [1], "role": "hyper_connection"}}], NATIVE)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "brewery.profile_preview.v1")
        resolved = first["resolved_recipe"]
        self.assertEqual(len(resolved["changes"]), 12)
        expected = sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":"))
                          .encode("ascii")).hexdigest()
        self.assertEqual(first["resolved_recipe_sha256"], expected)

    @unittest.skipUnless(NATIVE_AVAILABLE, "native plans not available; set BREWERY_PREVIEW_TEST_PLANS")
    def test_all_profile_differences_reproduce_other_native_plan_size(self):
        rules = [{"select": {"role": role}} for role in ("hyper_connection", "routed_expert")]
        for profile, expected in (("v2", 109716416292), ("ht43", 118785840932)):
            with self.subTest(profile=profile):
                report = self.preview(profile, rules, NATIVE)
                self.assertEqual(report["size"]["predicted_hgn_bytes"], expected)
                self.assertEqual(len(report["resolved_recipe"]["changes"]), 530)

    @unittest.skipUnless(NATIVE_AVAILABLE, "native plans not available; set BREWERY_PREVIEW_TEST_PLANS")
    def test_rejects_other_source_in_expert_rotation_companion(self):
        for profile in ("v2", "ht43"):
            report = json.loads((NATIVE / f"{profile}.json").read_text(encoding="utf-8"))
            if profile == "ht43":
                entries = {t["name"]: t for t in report["plan"]["files"][0]["tensors"]}
                entries[EXPERT.replace(".weight", ".svh")]["origin"]["HtScales"]["tensor"] = "other"
            (self.folder / f"{profile}.json").write_text(json.dumps(report), encoding="utf-8")
        self.reject("v2", [{"select": {"tensors": [EXPERT]}}], "incompatible rotation companion")


if __name__ == "__main__":
    unittest.main()
