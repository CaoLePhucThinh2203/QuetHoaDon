import unittest

from src.evaluation.reproduction import (compare_predictions, compare_scores, reference_predictions,
                                         score_predictions)

RECORDS = [
    {"img_id": "a.jpg", "gt": {"SELLER": ["MINIMART ANAN"], "TOTAL_COST": ["Tong tien:", "74,000"],
                               "TIMESTAMP": ["Ngay 11/08/2020"]}},
    {"img_id": "b.jpg", "gt": {"SELLER": ["CO.OPMART"], "TOTAL_COST": ["52,700"]}},
]
PREDS = {"a.jpg": {"SELLER": "MINIMART ANAN", "ADDRESS": None, "TIMESTAMP": "11/08/2020", "TOTAL_COST": "74,000"},
         "b.jpg": {"SELLER": "VINMART", "ADDRESS": None, "TIMESTAMP": None, "TOTAL_COST": "52,700"}}


def report_from(preds):
    s = score_predictions(RECORDS, preds)
    return {"split": "validation",
            "per_receipt_soft_rules": [{"img_id": k, "pred": v} for k, v in preds.items()],
            "fields_lenient_soft_rules": {f: {"f1": v} for f, v in s["fields"].items()},
            "receipt_exact_match_lenient_soft_rules": {"all_fields_correct": s["receipts"]["all_fields_correct"]}}


class TestReproduction(unittest.TestCase):
    def test_scores_follow_the_lenient_rules(self):
        s = score_predictions(RECORDS, PREDS)
        self.assertEqual(s["fields"]["TOTAL_COST"], 1.0)
        self.assertEqual(s["fields"]["TIMESTAMP"], 1.0)
        self.assertAlmostEqual(s["fields"]["SELLER"], 0.5)
        self.assertEqual(s["fields"]["ADDRESS"], 0.0)
        self.assertAlmostEqual(s["macro_f1"], (1.0 + 1.0 + 0.5 + 0.0) / 4)
        self.assertEqual(s["receipts"]["all_fields_correct"], 1)

    def test_identical_predictions_pass(self):
        rep = report_from(PREDS)
        cmp_p = compare_predictions(PREDS, reference_predictions(rep))
        self.assertEqual((cmp_p["receipts_compared"], cmp_p["receipts_identical"]), (2, 2))
        self.assertTrue(compare_scores(score_predictions(RECORDS, PREDS), rep)["identical"])

    def test_a_changed_field_is_reported(self):
        rep = report_from(PREDS)
        changed = {k: dict(v) for k, v in PREDS.items()}
        changed["a.jpg"]["TOTAL_COST"] = "7,000"
        cmp_p = compare_predictions(changed, reference_predictions(rep))
        self.assertEqual(cmp_p["receipts_identical"], 1)
        self.assertEqual(cmp_p["field_mismatches"]["TOTAL_COST"], 1)
        self.assertEqual(cmp_p["examples"][0], {"img_id": "a.jpg", "field": "TOTAL_COST", "new": "7,000",
                                                "reference": "74,000"})
        sc = compare_scores(score_predictions(RECORDS, changed), rep)
        self.assertFalse(sc["identical"])
        self.assertAlmostEqual(sc["difference"]["TOTAL_COST"], -0.5)

    def test_none_and_empty_string_are_the_same(self):
        cmp_p = compare_predictions({"a": {"SELLER": ""}}, {"a": {"SELLER": None}})
        self.assertEqual(cmp_p["receipts_identical"], 1)

    def test_missing_receipts_are_counted(self):
        cmp_p = compare_predictions({"a": {}}, {"a": {}, "b": {}})
        self.assertEqual((cmp_p["receipts_compared"], cmp_p["only_in_reference"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
