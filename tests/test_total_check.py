import tempfile
import unittest

from src.extraction.total_check import choose_total, format_amount, totals_from_items
from src.pipeline.extractor import ReceiptExtractor
from tests.test_extractor import make_meta, onehot, write_image
from tests.test_extractor_items import ITEMS
from tests.test_line_items import lay


def check(item_sum, printed, gross=None):
    return {"status": "verified" if item_sum in printed or gross in printed else "unverified", "sum": item_sum,
            "sum_without_discounts": gross if gross is not None else item_sum, "totals": printed}


class ChooseTotalTest(unittest.TestCase):
    def test_wrong_total_replaced_by_verified_item_sum(self):
        # the model picked the cash given ("tiền khách đưa"), which is not on a total line
        self.assertEqual(choose_total(150000, check(148000, [148000, 148000])), 148000)

    def test_missing_total_filled(self):
        self.assertEqual(choose_total(None, check(52000, [52000])), 52000)

    def test_total_printed_on_a_total_line_is_kept(self):
        # order-level discount: items add up to 78 000, the payable total 70 200 is also printed
        self.assertIsNone(choose_total(70200, check(78000, [78000, 70200, 70200])))
        self.assertIsNone(choose_total(52000, check(52000, [52000])))

    def test_unverified_items_never_change_the_total(self):
        self.assertIsNone(choose_total(150000, check(140000, [148000])))
        self.assertIsNone(choose_total(None, check(0, [])))

    def test_format(self):
        self.assertEqual(format_amount(585500), "585,500")


class TotalsFromItemsTest(unittest.TestCase):
    def test_predictions_are_updated_and_counted(self):
        words, boxes = lay(["Tên hàng SL Đơn giá Thành tiền", "Sữa tươi", "1 52,000 52,000", "Cộng tiền hàng 52,000",
                            "Tiền khách đưa 100,000"])
        records = [{"img_id": "a.jpg", "words": words, "boxes": boxes}]
        preds = {"a.jpg": {"TOTAL_COST": "100,000", "SELLER": "X"}}
        out, changed = totals_from_items(records, preds)
        self.assertEqual((out["a.jpg"]["TOTAL_COST"], out["a.jpg"]["SELLER"], changed), ("52,000", "X", 1))
        self.assertEqual(preds["a.jpg"]["TOTAL_COST"], "100,000")          # input untouched


class ExtractorTotalFromItemsTest(unittest.TestCase):
    def run_extractor(self, **kw):
        def tagger(rec, _):        # the model labels the first item amount "8,000" as the total
            return onehot(["B-TOTAL_COST" if w == "8,000" else "O" for w in rec["words"]])
        with tempfile.TemporaryDirectory() as tmp:
            return ReceiptExtractor(tagger, **kw).extract_from_ocr(make_meta(ITEMS), write_image(tmp))

    def test_off_by_default(self):
        total = self.run_extractor(items=True)["fields"]["TOTAL_COST"]
        self.assertEqual((total["amount_vnd"], total["source"]), (8000, "model"))

    def test_on(self):
        result = self.run_extractor(items=True, total_from_items=True)
        total = result["fields"]["TOTAL_COST"]
        self.assertEqual((total["text"], total["amount_vnd"], total["source"]), ("14,500", 14500, "items"))
        self.assertEqual(result["items_check"]["predicted_total"], 14500)

    def test_on_without_items_output(self):
        result = self.run_extractor(total_from_items=True)
        self.assertEqual(result["fields"]["TOTAL_COST"]["amount_vnd"], 14500)
        self.assertNotIn("items", result)


if __name__ == "__main__":
    unittest.main()
