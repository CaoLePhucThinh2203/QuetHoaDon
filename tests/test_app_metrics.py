import unittest

from src.evaluation.app_metrics import app_scores, extract_time, format_app_table, strict_text, wilson_interval

GTS = {
    "a": {"SELLER": ["MINIMART ANAN"], "ADDRESS": ["Địa chỉ: 12 Tran Phu"], "TIMESTAMP": ["Ngày: 11/08/2020 08:06"],
          "TOTAL_COST": ["Tổng tiền:", "74,000"]},
    "b": {"SELLER": ["CO.OPMART"], "TOTAL_COST": ["52,700"]},
    "c": {"TIMESTAMP": ["01/02/2021"], "TOTAL_COST": ["Tổng cộng"]},
}
PREDS = {
    "a": {"SELLER": "minimart  anan", "ADDRESS": "12 Tran Phu", "TIMESTAMP": "11/08/2020-08:06", "TOTAL_COST": "74,000"},
    "b": {"SELLER": "VINMART", "ADDRESS": None, "TIMESTAMP": None, "TOTAL_COST": "52.700"},
    "c": {"SELLER": None, "ADDRESS": None, "TIMESTAMP": "01/02/2021 10:30", "TOTAL_COST": None},
}


class TestHelpers(unittest.TestCase):
    def test_strict_text_prefix_only_for_addresses(self):
        self.assertEqual(strict_text("ADDRESS", "ĐC: 12 Tran  Phu"), "12 tran phu")
        self.assertEqual(strict_text("ADDRESS", "Địa chỉ 12 Tran Phu"), "12 tran phu")
        self.assertEqual(strict_text("SELLER", "Địa chỉ ABC"), "địa chỉ abc")
        self.assertEqual(strict_text("SELLER", None), "")

    def test_extract_time(self):
        self.assertEqual(extract_time("Giờ: 16:49"), "16:49")
        self.assertEqual(extract_time("17h23"), "17:23")
        self.assertEqual(extract_time("22/01/2020 Giờ: 16:49"), "16:49")
        self.assertEqual(extract_time("11/08/2020-08:06"), "08:06")
        self.assertEqual(extract_time("14/08/2 2020 08:06"), "08:06")
        self.assertIsNone(extract_time("22/01/2020"))
        self.assertIsNone(extract_time(None))

    def test_wilson_interval(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)


class TestAppScores(unittest.TestCase):
    def test_counts(self):
        s = app_scores(GTS, PREDS)["fields"]
        self.assertEqual((s["SELLER"]["n"], s["SELLER"]["correct"]), (2, 1))
        self.assertEqual((s["ADDRESS"]["n"], s["ADDRESS"]["correct"]), (1, 1))
        self.assertEqual((s["TIMESTAMP"]["n"], s["TIMESTAMP"]["correct"]), (2, 2))
        self.assertEqual((s["TIMESTAMP"]["time"]["n"], s["TIMESTAMP"]["time"]["correct"]), (1, 1))
        self.assertEqual((s["TOTAL_COST"]["n"], s["TOTAL_COST"]["correct"]), (2, 2))
        self.assertGreater(s["SELLER"]["mean_cer"], 0.0)

    def test_all_correct(self):
        a = app_scores(GTS, PREDS)["receipts_all_correct"]
        self.assertEqual((a["n"], a["correct"]), (3, 2))

    def test_wrong_time_breaks_the_receipt_but_not_the_date(self):
        preds = {k: dict(v) for k, v in PREDS.items()}
        preds["a"]["TIMESTAMP"] = "11/08/2020 09:06"
        s = app_scores(GTS, preds)
        self.assertEqual(s["fields"]["TIMESTAMP"]["correct"], 2)
        self.assertEqual(s["fields"]["TIMESTAMP"]["time"]["correct"], 0)
        self.assertEqual(s["receipts_all_correct"]["correct"], 1)

    def test_missing_predictions_are_counted(self):
        s = app_scores(GTS, {})
        self.assertEqual(s["fields"]["SELLER"]["no_prediction"], 2)
        self.assertEqual(s["fields"]["TOTAL_COST"]["no_prediction"], 2)
        self.assertEqual(s["fields"]["SELLER"]["correct"], 0)
        self.assertEqual(s["receipts_all_correct"]["correct"], 0)

    def test_total_needs_a_plausible_annotated_amount(self):
        gts = {"x": {"TOTAL_COST": ["8936100000017"]}}                  # a barcode, not a total
        s = app_scores(gts, {"x": {"TOTAL_COST": "74,000"}})
        self.assertEqual(s["fields"]["TOTAL_COST"]["n"], 0)

    def test_table(self):
        text = format_app_table(app_scores(GTS, PREDS), "M1b")
        self.assertIn("### M1b", text)
        self.assertIn("| SELLER | 1 / 2 |", text)
        self.assertIn("**All fields correct** | 2 / 3", text)
        self.assertIn("TIMESTAMP time hh:mm | 1 / 1", text)


if __name__ == "__main__":
    unittest.main()
