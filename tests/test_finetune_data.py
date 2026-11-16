import unittest
from types import SimpleNamespace

import numpy as np

from src.recognition.finetune_data import (RegionPairConfig, align_segments, crop_with_margin, in_vocab, padded_view,
                                           receipt_pairs)


class TestAlignSegments(unittest.TestCase):
    def test_docstring_example(self):
        self.assertEqual(align_segments(["Tong", "tien:74,0OO"], "Tổng tiền: 74,000"), ["Tổng", "tiền: 74,000"])

    def test_one_part_takes_everything(self):
        self.assertEqual(align_segments(["abc"], " Cửa hàng ABC "), ["Cửa hàng ABC"])

    def test_three_parts_with_reading_errors(self):
        parts = ["12", "Tran", "Phu"]
        self.assertEqual(align_segments(parts, "12 Trần Phú"), ["12", "Trần", "Phú"])

    def test_diacritics_only(self):
        self.assertEqual(align_segments(["Ngay:", "11/08/2020"], "Ngày: 11/08/2020"), ["Ngày:", "11/08/2020"])

    def test_cut_snaps_to_a_space(self):
        # the OCR read one extra character, the cut must still fall on the space of the annotation
        self.assertEqual(align_segments(["Q.1x", "HCM"], "Q.1 HCM"), ["Q.1", "HCM"])

    def test_empty(self):
        self.assertEqual(align_segments([], "abc"), [])


def instance(text, label, x, y, w, h, size=(1000, 1000)):
    return SimpleNamespace(text=text, label=label, segmentation=[[x, y, x + w, y, x + w, y + h, x, y + h]],
                           raw_bbox=[x, y, w, h], width=size[0], height=size[1])


def meta(items, size=(1000, 1000)):
    return {"orig_shape": [size[1], size[0], 3], "processed_shape": [size[1], size[0], 3], "skew_angle": 0.0,
            "ocr_items": [{"text": t, "bbox": b} for t, b in items]}


class TestReceiptPairs(unittest.TestCase):
    def test_line_and_region_pairs(self):
        inst = [instance("Tổng tiền: 74,000", "TOTAL_COST", 100, 100, 400, 30)]
        m = meta([("Tong", [105, 102, 180, 128]), ("tien:74,0OO", [190, 102, 490, 128]), ("x", [600, 600, 650, 630])])
        pairs, st = receipt_pairs(m, inst)
        self.assertEqual([(p["kind"], p["text"]) for p in pairs],
                         [("line", "Tổng tiền: 74,000"), ("region", "Tổng"), ("region", "tiền: 74,000")])
        self.assertEqual(st["lines"], 1)
        self.assertEqual(st["regions"], 2)

    def test_line_far_from_ocr_keeps_only_the_line(self):
        inst = [instance("Cửa hàng Hoàng Long", "SELLER", 100, 100, 400, 30)]
        m = meta([("zzzz qqqq", [105, 102, 490, 128])])
        pairs, st = receipt_pairs(m, inst, RegionPairConfig(max_line_cer=0.5))
        self.assertEqual([p["kind"] for p in pairs], ["line"])
        self.assertEqual(st["line_skipped_cer"], 1)

    def test_box_mostly_outside_is_ignored(self):
        inst = [instance("ABC", "SELLER", 100, 100, 100, 30)]
        m = meta([("ABC", [150, 100, 300, 130])])                 # only a third inside
        pairs, _ = receipt_pairs(m, inst)
        self.assertEqual([p["kind"] for p in pairs], ["line"])

    def test_frame_mismatch(self):
        inst = [instance("ABC", "SELLER", 1, 1, 10, 10, size=(500, 1000))]
        pairs, st = receipt_pairs(meta([]), inst)
        self.assertEqual(pairs, [])
        self.assertEqual(st["frame_mismatch"], 1)


class TestCrops(unittest.TestCase):
    def test_margin_and_padded_view(self):
        img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        crop, inner = crop_with_margin(img, [50, 40, 90, 60], margin=8)
        self.assertEqual(crop.shape[:2], (36, 56))
        self.assertEqual(inner, [8, 8, 48, 28])
        view = padded_view(crop, inner, 3, 2)
        self.assertTrue((view == img[38:62, 47:93]).all())

    def test_margin_is_clipped_at_the_border(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        crop, inner = crop_with_margin(img, [2, 3, 20, 10], margin=8)
        self.assertEqual(inner[:2], [2, 3])
        self.assertEqual(crop.shape[:2], (18, 28))

    def test_in_vocab(self):
        self.assertTrue(in_vocab("ab c", "abc "))
        self.assertFalse(in_vocab("ab€", "abc "))


if __name__ == "__main__":
    unittest.main()
