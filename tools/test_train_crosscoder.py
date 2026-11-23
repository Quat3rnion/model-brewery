import base64
import hashlib
import struct
import unittest
import urllib.parse

import torch

from train_crosscoder import W1_EXECUTABLE, capture_pair, codes, fit


class TrainCrosscoderTest(unittest.TestCase):
    def test_one_topk_code_reconstructs_both_heldout_sides(self):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(5)
        sample = torch.randn((384, 2), generator=rng)
        pairs = torch.stack((sample + 3, sample.flip(1) * 2 - 1), dim=1)
        weights, scale, metrics = fit(pairs[:320], pairs[320:], features=16, k=3, steps=250)

        active = codes(pairs[320:], weights, scale, 3)
        self.assertEqual(active.shape, (64, 16))
        self.assertLessEqual(int(active.count_nonzero(dim=1).max()), 3)
        reconstruction = (torch.einsum("nf,sfd->nsd", active, weights["W_dec"])
                          + weights["b_dec"]) / scale
        for index, side in enumerate(("parent", "child")):
            self.assertGreater(metrics[side]["explained_variance"], 0.9)
            self.assertLess(metrics[side]["dead_features"], 16)
            self.assertAlmostEqual(metrics[side]["mse"],
                                   float((pairs[320:, index] - reconstruction[:, index]).square().mean()),
                                   places=6)

    def test_reject_misaligned_shapes(self):
        with self.assertRaisesRegex(ValueError, "paired"):
            fit(torch.ones(5, 2, 3), torch.ones(5, 1, 3), features=5, k=2, steps=1)

    def test_trace_rows_require_matching_tokens_and_verified_payloads(self):
        ids = ["%08x-0000-0000-0000-000000000000" % i for i in (1, 2)]
        pair = dict(parent_job_id=ids[0], child_job_id=ids[1], split="train")
        fault = None

        def get(path):
            path = urllib.parse.urlsplit(path).path
            i = ids.index(path.split("/")[3])
            sha = "ab"[i] * 64
            receipt = {"kind": "trace", "schema_version": 1, "model_id": ids[i], "model_sha256": sha,
                       "executable_sha256": W1_EXECUTABLE, "prompts": [{"manifest_sha256": sha}]}
            if path.endswith("/trace"):
                return {"schema_version": 1, "prompt": 0, "model_id": ids[i], "model_sha256": sha,
                        "manifest_sha256": sha, "intervention_count": 0,
                        "spec": {"positions": [0], "layers": [1], "roles": ["l_out"]},
                        "tokens": [{"position": 0, "token_id": 123 + i * (fault == "token")}]}
            if path.endswith("/tensor"):
                raw = struct.pack("<ff", i + 1, i + 2)
                return {"prompt": 0, "layer": 1, "role": "l_out", "dtype": "f32_le",
                        "missing_positions": [], "rows": [
                            {"position": 0, "shape": [2], "data_base64": base64.b64encode(raw).decode(),
                             "sha256": "0" * 64 if fault == "digest" else hashlib.sha256(raw).hexdigest()}]}
            return {"kind": "trace", "state": "succeeded", "input_ids": [ids[i]], "options": {},
                    "prompt_text": "same text", "result": receipt}

        rows, revisions, labels = capture_pair(get, pair, 1, "l_out")
        self.assertEqual(rows.tolist(), [[[1, 2], [2, 3]]])
        self.assertEqual(revisions["child"], {"id": ids[1], "sha256": "b" * 64})
        self.assertEqual(labels, [{"prompt": "same text", "position": 0, "token_id": 123}])
        fault = "token"
        with self.assertRaisesRegex(ValueError, "tokens"):
            capture_pair(get, pair, 1, "l_out")
        fault = "digest"
        with self.assertRaisesRegex(ValueError, "verified"):
            capture_pair(get, pair, 1, "l_out")
