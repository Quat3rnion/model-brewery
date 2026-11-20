import os
from pathlib import Path
import unittest

from tools import glq_ref


class GlqReferenceTests(unittest.TestCase):
    def test_glq_label_alone_does_not_identify_trellis(self):
        for codebook, variant, layout in (
            ("e8_shell", "3inst", "kernel"),
            ("trellis", "hyb", "kernel"),
            ("trellis", "3inst", "natural"),
        ):
            with self.subTest(codebook=codebook, variant=variant, layout=layout):
                config = {"quantization_config": {
                    "quant_method": "glq", "codebook": codebook,
                    "variant": variant, "trellis_layout": layout,
                    "codesz": 16, "bpw": 4, "block_diagonal": True,
                }}
                with self.assertRaisesRegex(ValueError, "3INST kernel"):
                    glq_ref.read_recipe(config)

    @unittest.skipUnless(os.environ.get("GLQ_REF_SOURCE"), "set GLQ_REF_SOURCE to pinned GLQ source")
    def test_cpu_native_encode_then_read_serialized_bytes(self):
        source = Path(os.environ["GLQ_REF_SOURCE"])
        for bpw, expected in ((2, [2]), (3, [3]), (4, [4]), (5, [4, 1]),
                              (6, [4, 2]), (7, [4, 3]), (8, [4, 4])):
            with self.subTest(bpw=bpw):
                evidence = glq_ref.probe(source, bpw)
                self.assertEqual(evidence["shape"], [32, 64])
                self.assertEqual(evidence["stage_bits"], expected)
                self.assertEqual(evidence["packed_bytes"], [256 * k for k in expected])
                self.assertEqual(evidence["normalized_decode"], "bit_exact")
                self.assertEqual(evidence["restored_decode"], "matches_encoder")
                self.assertEqual(evidence["dropped_residual_refused"], len(expected) == 2)
