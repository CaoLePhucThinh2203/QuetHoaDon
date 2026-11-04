import unittest

from src.evaluation import (cer, corpus_error_rate, detection_scores, field_match, field_scores,
                            group_gt_fields, normalize_amount, normalize_text, prf1, wer)
from src.preprocessing.annotation_parser import AnnotationInstance


class TestOcrErrors(unittest.TestCase):
    def test_cer_wer(self):
        self.assertAlmostEqual(cer("abc", "abd"), 1 / 3)
        self.assertAlmostEqual(wer("a b c", "a x c"), 1 / 3)
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "x"), 1.0)

    def test_corpus_rate_is_weighted_not_mean_of_ratios(self):
        pairs = [("abcdefghij", "abcdefghij"), ("ab", "xx")]   # 2 edits / 12 chars
        self.assertAlmostEqual(corpus_error_rate(pairs), 2 / 12)


class TestDetection(unittest.TestCase):
    def test_scores(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        pred = [[0, 0, 10, 10], [100, 100, 110, 110]]
        s = detection_scores(gt, pred, 0.5)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 1, 1))
        self.assertAlmostEqual(s["recall"], 0.5)

    def test_one_to_one_matching(self):
        gt = [[0, 0, 10, 10]]
        pred = [[0, 0, 10, 10], [0, 0, 10, 11]]
        self.assertEqual(detection_scores(gt, pred)["tp"], 1)

    def test_prf1_zero_division(self):
        self.assertEqual(prf1(0, 0, 0)["f1"], 0.0)


class TestNormalisation(unittest.TestCase):
    def test_amounts(self):
        cases = {"27,000": 27000, "27.000đ": 27000, "100,000.00": 100000, "1.250.000 VND": 1250000,
                 "TỔNG: 73,000": 73000, "abc": None, "": None, None: None}
        for text, expected in cases.items():
            self.assertEqual(normalize_amount(text), expected, text)

    def test_text(self):
        self.assertEqual(normalize_text("  Cửa   Hàng "), "cửa hàng")
        self.assertEqual(normalize_text("Cửa Hàng Đẹp", ignore_diacritics=True), "cua hang dep")


class TestFields(unittest.TestCase):
    def test_total_matches_any_gt_region_amount(self):
        self.assertTrue(field_match("TOTAL_COST", "73.000", ["TONG CONG", "73,000"]))
        self.assertFalse(field_match("TOTAL_COST", "72,000", ["73,000"]))

    def test_text_field_joined_regions(self):
        self.assertTrue(field_match("ADDRESS", "12 le loi q1", ["12 Le Loi", "Q1"]))
        self.assertFalse(field_match("ADDRESS", None, ["x"]))

    def test_field_scores(self):
        gts = {"a": {"SELLER": ["shop a"], "TOTAL_COST": ["10,000"]},
               "b": {"SELLER": ["shop b"], "TOTAL_COST": ["20,000"]}}
        preds = {"a": {"SELLER": "Shop A", "TOTAL_COST": "10.000"},
                 "b": {"SELLER": "wrong", "TOTAL_COST": None}}
        sc = field_scores(preds, gts, fields=("SELLER", "TOTAL_COST"))
        self.assertEqual((sc["SELLER"]["tp"], sc["SELLER"]["fp"], sc["SELLER"]["fn"]), (1, 1, 1))
        self.assertAlmostEqual(sc["SELLER"]["exact_match"], 0.5)
        self.assertEqual((sc["TOTAL_COST"]["tp"], sc["TOTAL_COST"]["fn"]), (1, 1))

    def test_group_gt_fields_orders_top_to_bottom(self):
        def inst(text, label, x, y):
            return AnnotationInstance(text, label, 0, [x, y, 10, 10], [0, 0, 0, 0], [], 0.0, 100, 100)
        g = group_gt_fields([inst("line 2", "ADDRESS", 0, 50), inst("line 1", "ADDRESS", 0, 10)])
        self.assertEqual(g["ADDRESS"], ["line 1", "line 2"])


if __name__ == "__main__":
    unittest.main()
