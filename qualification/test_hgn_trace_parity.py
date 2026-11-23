"""Acceptance boundaries for comparison against Halogen rows."""
import unittest

from hgn_trace_parity import parity_passes


class ParityGateTest(unittest.TestCase):
    def test_gate_requires_every_miss_to_be_a_near_tie(self):
        self.assertTrue(parity_passes(62, 0.01, [0.298, 0.700]))
        self.assertTrue(parity_passes(64, 0.01, []))
        self.assertFalse(parity_passes(62, 0.009392, [0.298, 1.0]))
        self.assertFalse(parity_passes(63, 0.009392, [1.01]))
        self.assertFalse(parity_passes(61, 0.009392, [0.1, 0.2, 0.3]))
        self.assertFalse(parity_passes(64, 0.010001, []))
