import unittest

import torch

from train_sae import fit


class TrainSaeTest(unittest.TestCase):
    def test_reconstruct_heldout_rows_and_export_decoder_orientation(self):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(5)
        sample = torch.randn((384, 3), generator=rng) * torch.tensor([2., 0.5, 0.2]) + 3
        weights, normalization, metrics = fit(sample[:320], sample[320:], features=16, k=3, steps=220)
        self.assertGreater(metrics["explained_variance"], 0.9)
        self.assertLess(metrics["mse"], 0.1)
        self.assertLessEqual(metrics["l0"], 3)
        self.assertGreater(metrics["l0"], 0)
        self.assertLess(metrics["dead_features"], 16)
        self.assertEqual(weights["W_enc"].shape, (3, 16))
        self.assertEqual(weights["W_dec"].shape, (16, 3))
        x = sample[320:] * normalization["scale"] - weights["b_dec"]
        strengths, ids = (x @ weights["W_enc"] + weights["b_enc"]).relu().topk(3, dim=-1)
        reconstructed = ((strengths[..., None] * weights["W_dec"][ids]).sum(dim=1)
                         + weights["b_dec"]) / normalization["scale"]
        self.assertAlmostEqual(metrics["mse"], float((sample[320:] - reconstructed).square().mean()), places=6)
