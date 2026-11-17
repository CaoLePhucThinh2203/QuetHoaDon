import unittest

import numpy as np

from src.recognition.finetune import (augment, cer_summary, lr_factor, per_receipt_edits, resized_width,
                                      target_tensors, width_buckets)


class TestResizedWidth(unittest.TestCase):
    def test_same_rule_as_vietocr(self):
        self.assertEqual(resized_width(100, 32), 100)
        self.assertEqual(resized_width(101, 32), 110)            # rounded up to a multiple of 10
        self.assertEqual(resized_width(10, 32), 32)              # minimum width
        self.assertEqual(resized_width(5000, 32), 512)           # maximum width


class TestBuckets(unittest.TestCase):
    def test_same_width_inside_a_batch_and_nothing_dropped(self):
        widths = [100, 110, 100, 100, 110, 120, 100]
        batches = width_buckets(widths, 2, np.random.default_rng(0))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(7)))
        for b in batches:
            self.assertEqual(len({widths[i] for i in b}), 1)
            self.assertLessEqual(len(b), 2)

    def test_deterministic_without_rng(self):
        self.assertEqual(width_buckets([20, 10, 20], 8), [[1], [0, 2]])


class TestTargets(unittest.TestCase):
    def test_like_the_vietocr_collator(self):
        ti, to, mask = target_tensors([[1, 5, 6, 2], [1, 7, 2]])
        self.assertEqual(ti.shape, (4, 2))
        self.assertEqual(ti[:, 1].tolist(), [1, 7, 2, 0])
        self.assertEqual(to.tolist(), [[5, 6, 2, 0], [7, 2, 0, 0]])
        self.assertEqual(mask.tolist(), [[False, False, False, True], [False, False, True, True]])


class TestMisc(unittest.TestCase):
    def test_cer_summary(self):
        s = cer_summary(["abcd", "Xy"], ["abed", "xy"])
        self.assertAlmostEqual(s["cer"], 1 / 6)
        self.assertEqual(s["exact"], 0.5)

    def test_per_receipt_edits(self):
        rows = [{"img_id": "a", "text": "ab"}, {"img_id": "a", "text": "cd"}, {"img_id": "b", "text": "x"}]
        self.assertEqual(per_receipt_edits(rows, ["ab", "ce", ""]), {"a": (1, 4), "b": (1, 1)})

    def test_lr_factor(self):
        self.assertLess(lr_factor(0, 100), lr_factor(4, 100))
        self.assertAlmostEqual(lr_factor(5, 100), 1.0, places=2)
        self.assertAlmostEqual(lr_factor(10 ** 6, 100), 0.05)

    def test_augment_keeps_shape_and_type(self):
        img = np.full((32, 80, 3), 200, dtype=np.uint8)
        out = augment(img, np.random.default_rng(1))
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
