import importlib.util
import unittest
from pathlib import Path

from src.recognition.quality import blank_unreadable

_spec = importlib.util.spec_from_file_location(
    "score_items", Path(__file__).resolve().parent.parent / "scripts" / "score_items.py")
score_items = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_items)

INVENTED = ("Tổng thanh toán 10.000đ " * 8).split()          # 24 words, one phrase repeated: unreadable
NORMAL = "Phúc Anh Minimart 248 Cổ Bi Gia Lâm Sữa TH 1 238,000 238,000 Tổng cộng 238,000 Tiền mặt 300,000".split()


def record(img_id, words):
    boxes = [[10 * i, 10, 10 * i + 8, 20] for i in range(len(words))]
    return {"img_id": img_id, "words": words, "boxes": boxes, "gt": {"TOTAL_COST": ["238,000"]}}


class BlankUnreadableTest(unittest.TestCase):
    def test_flagged_receipt_loses_every_prediction(self):
        recs = [record("bad", INVENTED), record("good", NORMAL)]
        preds = {"bad": {"TOTAL_COST": "10.000"}, "good": {"TOTAL_COST": "238,000", "SELLER": "Phúc Anh"}}
        out, flagged = blank_unreadable(recs, preds)
        self.assertEqual(flagged, ["bad"])
        self.assertEqual(out["bad"], {})
        self.assertEqual(out["good"], preds["good"])
        self.assertEqual(preds["bad"], {"TOTAL_COST": "10.000"})        # input untouched

    def test_nothing_flagged(self):
        out, flagged = blank_unreadable([record("good", NORMAL)], {"good": {"SELLER": "x"}})
        self.assertEqual(flagged, [])
        self.assertEqual(out, {"good": {"SELLER": "x"}})


class ScoreItemsQualityTest(unittest.TestCase):
    def test_flagged_receipt_shows_no_item(self):
        recs = [record("bad", INVENTED), record("good", NORMAL)]
        plain = score_items.score_records(recs, n_examples=0)
        checked = score_items.score_records(recs, n_examples=0, quality_check=True)
        self.assertEqual(plain["flagged_unreadable"], 0)
        self.assertEqual(checked["flagged_unreadable"], 1)
        self.assertEqual(checked["receipts_with_gold_total"], 2)        # still counted, as not read
        self.assertLessEqual(checked["found_items"]["count"], plain["found_items"]["count"])
        self.assertLessEqual(checked["verified"]["count"], plain["verified"]["count"])
        bad_rows = [r for r in checked["failures"] + checked["examples"] if r["img_id"] == "bad"]
        self.assertEqual(bad_rows, [])                                   # no item read for the flagged receipt


if __name__ == "__main__":
    unittest.main()
