import math
import unittest

import numpy as np

from src.preprocessing.token_dataset import LABEL2ID
from src.training import field_eval as fe


def onehot(tags):
    probs = np.zeros((len(tags), len(LABEL2ID)))
    for i, t in enumerate(tags):
        probs[i, LABEL2ID[t]] = 1.0
    return probs


R1_TAGS = ["B-SELLER", "I-SELLER", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST"]


def rec1():
    return {"img_id": "a.jpg", "file_id": "a", "words": ["MINI", "MART", "Tong", "tien:", "74,000"],
            "labels": R1_TAGS, "gt": {"SELLER": ["MINI MART"], "TOTAL_COST": ["Tong tien:", "74,000"]}}


def rec2():
    return {"img_id": "b.jpg", "file_id": "b", "words": ["Tong", "52,000"],
            "labels": ["B-TOTAL_COST", "I-TOTAL_COST"], "gt": {"TOTAL_COST": ["Tong", "50,000"]}}


class TestFieldEval(unittest.TestCase):
    def test_perfect_prediction(self):
        recs = [rec1()]
        merged = [{"probs": onehot(R1_TAGS)}]
        preds = fe.decode_predictions(recs, merged)
        self.assertEqual(preds["a.jpg"]["SELLER"], "MINI MART")
        self.assertEqual(preds["a.jpg"]["TOTAL_COST"], "74,000")
        rep = fe.build_report(recs, merged)
        self.assertEqual(rep["fields"]["SELLER"]["f1"], 1.0)
        self.assertEqual(rep["fields"]["TOTAL_COST"]["f1"], 1.0)
        self.assertEqual(rep["receipt_exact_match"]["rate"], 1.0)
        self.assertEqual(rep["total_amount"]["rmse"], 0.0)
        self.assertEqual(rep["total_cost_errors"], [])
        self.assertEqual(rep["token_auc"]["B-SELLER"], 1.0)
        self.assertIsNone(rep["token_auc"]["B-ADDRESS"])        # absent from the gold tags
        self.assertEqual(sum(sum(row) for row in rep["confusion"]), 5)

    def test_missing_amount_is_a_miss_not_a_crash(self):
        recs = [rec1()]
        tags = ["B-SELLER", "I-SELLER", "B-TOTAL_COST", "I-TOTAL_COST", "O"]
        rep = fe.build_report(recs, [{"probs": onehot(tags)}])
        self.assertEqual(rep["fields"]["TOTAL_COST"]["recall"], 0.0)
        self.assertEqual(rep["total_amount"]["no_amount_predicted"], 1)
        self.assertIsNone(rep["total_amount"]["rmse"])
        self.assertEqual(rep["receipt_exact_match"]["rate"], 0.0)
        self.assertEqual(len(rep["total_cost_errors"]), 1)
        self.assertIsNone(rep["total_cost_errors"][0]["predicted"])

    def test_amount_error_statistics(self):
        recs = [rec1(), rec2()]
        merged = [{"probs": onehot(R1_TAGS)}, {"probs": onehot(["B-TOTAL_COST", "I-TOTAL_COST"])}]
        a = fe.build_report(recs, merged)["total_amount"]
        self.assertEqual((a["compared"], a["exact"]), (2, 1))
        self.assertAlmostEqual(a["mae"], 1000.0)
        self.assertAlmostEqual(a["rmse"], math.sqrt(2_000_000))
        self.assertAlmostEqual(a["median_abs_error"], 1000.0)

    def test_gt_amount_takes_the_largest_number(self):
        self.assertEqual(fe.gt_amount(["Tổng tiền:", "74,000"]), 74000)
        self.assertEqual(fe.gt_amount(["Tổng tiền:"]), None)
        self.assertIsNone(fe.gt_amount(None))

    def test_token_prf_from_confusion(self):
        recs = [rec1()]
        tags = ["B-SELLER", "O", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST"]      # one word wrong
        cm = fe.word_confusion(recs, [{"probs": onehot(tags)}])
        prf = fe.token_prf(cm)
        self.assertEqual(prf["I-SELLER"]["recall"], 0.0)
        self.assertEqual(prf["B-SELLER"]["f1"], 1.0)
        self.assertEqual(int(cm[LABEL2ID["I-SELLER"], LABEL2ID["O"]]), 1)


def probs_row(**mass):
    """One word's probability row from {label: probability}; the rest goes to O."""
    row = np.zeros(len(LABEL2ID))
    for lab, p in mass.items():
        row[LABEL2ID[lab.replace("__", "-")]] = p
    row[0] = 1.0 - row.sum()
    return row


class TestSoftDecoding(unittest.TestCase):
    def test_soft_total_picks_the_amount_with_most_total_mass_even_if_argmax_is_o(self):
        words = ["Tong", "tien", "74,000", "5,000"]
        probs = np.stack([probs_row(), probs_row(), probs_row(I__TOTAL_COST=0.4), probs_row(I__TOTAL_COST=0.01)])
        self.assertEqual(fe.soft_total(words, probs), "74,000")

    def test_soft_total_returns_none_without_evidence_or_plausible_amount(self):
        probs = np.stack([probs_row(), probs_row()])
        self.assertIsNone(fe.soft_total(["74,000", "x"], probs))                      # no TOTAL mass at all
        probs = np.stack([probs_row(I__TOTAL_COST=0.9)])
        self.assertIsNone(fe.soft_total(["0902543800123"], probs))                    # phone-number-like
        self.assertIsNone(fe.soft_total(["500"], probs))                              # below 1000 VND

    def test_soft_timestamp(self):
        words = ["Ngay", "14/08/2020", "16:22"]
        probs = np.stack([probs_row(), probs_row(I__TIMESTAMP=0.5), probs_row(I__TIMESTAMP=0.5)])
        self.assertEqual(fe.soft_timestamp(words, probs), "14/08/2020")
        self.assertIsNone(fe.soft_timestamp(["abc"], np.stack([probs_row(I__TIMESTAMP=0.9)])))

    def test_normalize_date(self):
        self.assertEqual(fe.normalize_date("14/08/2020"), "14/08/2020")
        self.assertEqual(fe.normalize_date("1-8-20"), "01/08/2020")
        self.assertEqual(fe.normalize_date("Ngày bán:14/08/2020 16:22"), "14/08/2020")
        self.assertIsNone(fe.normalize_date("khong co ngay"))
        self.assertIsNone(fe.normalize_date(None))


class TestLenientMatch(unittest.TestCase):
    def test_seller_tolerates_small_ocr_errors(self):
        self.assertTrue(fe.lenient_match("SELLER", "MINIMART ANAM", ["MINIMART ANAN"]))
        self.assertFalse(fe.lenient_match("SELLER", "Cua hang khac", ["MINIMART ANAN"]))

    def test_timestamp_compares_dates_only(self):
        self.assertTrue(fe.lenient_match("TIMESTAMP", "11/08/2020", ["Ngày : 11/08/2020 08:06"]))
        self.assertFalse(fe.lenient_match("TIMESTAMP", "12/08/2020", ["Ngày : 11/08/2020 08:06"]))

    def test_total_uses_the_amount_rule_and_empty_never_matches(self):
        self.assertTrue(fe.lenient_match("TOTAL_COST", "74,000", ["Tổng tiền:", "74,000"]))
        self.assertFalse(fe.lenient_match("TOTAL_COST", None, ["74,000"]))
        self.assertFalse(fe.lenient_match("SELLER", "abc", []))

    def test_report_contains_the_new_sections(self):
        recs = [rec1()]
        rep = fe.build_report(recs, [{"probs": onehot(R1_TAGS)}])
        for key in ("fields_lenient_hard", "fields_lenient_soft", "receipt_exact_match_lenient_soft", "total_amount_soft"):
            self.assertIn(key, rep)
        self.assertEqual(rep["fields_lenient_soft"]["TOTAL_COST"]["f1"], 1.0)
        self.assertEqual(rep["receipt_exact_match_lenient_soft"]["rate"], 1.0)


class TestTotalDiagnostics(unittest.TestCase):
    @staticmethod
    def rec(name, words, gt_total):
        gt = {"TOTAL_COST": gt_total} if gt_total else {"SELLER": ["x"]}
        return {"img_id": name + ".jpg", "file_id": name, "words": words, "labels": ["O"] * len(words), "gt": gt}

    def test_every_cause_is_counted(self):
        recs = [
            self.rec("top", ["Tong", "74,000", "5,000"], ["Tong", "74,000"]),
            self.rec("lower", ["74,000", "5,000"], ["74,000"]),
            self.rec("absent", ["Tong", "5,000"], ["74,000"]),
            self.rec("nogt", ["a"], None),
            self.rec("weak", ["74,000"], ["74,000"]),
            self.rec("pair", ["52", "700"], ["52,700"]),
        ]
        merged = [
            {"probs": np.stack([probs_row(), probs_row(I__TOTAL_COST=0.5), probs_row(I__TOTAL_COST=0.1)])},
            {"probs": np.stack([probs_row(I__TOTAL_COST=0.1), probs_row(I__TOTAL_COST=0.6)])},
            {"probs": np.stack([probs_row(), probs_row(I__TOTAL_COST=0.3)])},
            {"probs": np.stack([probs_row()])},
            {"probs": np.stack([probs_row()])},
            {"probs": np.stack([probs_row(I__TOTAL_COST=0.5), probs_row()])},
        ]
        d = fe.total_diagnostics(recs, merged)
        self.assertEqual(d["categories"], {"no_gt_amount": 1, "amount_not_in_ocr": 1, "below_min_prob": 1,
                                           "ranked_lower": 1, "top1": 2})
        self.assertEqual(d["examples"]["ranked_lower"], ["lower"])
        self.assertEqual(sorted(d["examples"]["top1"]), ["pair", "top"])

    def test_report_carries_the_diagnosis(self):
        rep = fe.build_report([rec1()], [{"probs": onehot(R1_TAGS)}])
        self.assertEqual(rep["total_diagnostics"]["categories"]["top1"], 1)


if __name__ == "__main__":
    unittest.main()
