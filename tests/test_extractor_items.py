import tempfile
import unittest

from src.pipeline.extractor import ReceiptExtractor, compact_result
from tests.test_extractor import make_meta, onehot, write_image

# a receipt with a header row, two items and a total line (boxes in the 1000 x 1000 pixel frame of make_meta)
ITEMS = [("MINIMART ANAN", [100, 50, 500, 90]),
         ("Ngay 11/08/2020 16:02", [100, 100, 700, 140]),
         ("Ten hang SL Don gia Thanh tien", [100, 200, 900, 240]),
         ("Bim bim Oishi 48g", [100, 300, 500, 340]),
         ("2 4,000 8,000", [100, 350, 500, 390]),
         ("Sua Moc Chau 180ml", [100, 400, 500, 440]),
         ("1 6,500 6,500", [100, 450, 500, 490]),
         ("Tong cong: 14,500", [100, 800, 600, 840])]
WORDS = " ".join(t for t, _ in ITEMS).split()


def tags_for(words):
    out = []
    for w in words:
        out.append("O")
    return out


class TestItems(unittest.TestCase):
    def run_extractor(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            image = write_image(tmp)
            return ReceiptExtractor(lambda rec, d: onehot(tags_for(rec["words"])), **kw).extract_from_ocr(
                make_meta(ITEMS), image)

    def test_items_are_off_by_default(self):
        result = self.run_extractor()
        self.assertNotIn("items", result)

    def test_items_and_check(self):
        result = self.run_extractor(items=True)
        self.assertEqual([(i["name"], i["quantity"], i["amount"]) for i in result["items"]],
                         [("Bim bim Oishi 48g", 2, 8000), ("Sua Moc Chau 180ml", 1, 6500)])
        self.assertEqual(result["items_check"]["sum"], 14500)
        self.assertEqual(result["items_check"]["status"], "verified")

    def test_compact_hides_fields_that_were_not_read(self):
        result = self.run_extractor(items=True)
        compact = compact_result(result)
        self.assertNotIn("SELLER", compact["fields"])         # the stub tagger labelled everything O
        self.assertEqual(len(compact["items"]), 2)
        self.assertEqual(compact["items_check"]["status"], "verified")
        self.assertNotIn("words", compact)

    def test_compact_has_time_and_amount(self):
        full = {"file_id": "x", "fields": {
            "TIMESTAMP": {"text": "Ngay 11/08/2020 16:02", "date": "11/08/2020"},
            "TOTAL_COST": {"text": "74,000", "amount_vnd": 74000}, "SELLER": {"text": None},
            "ADDRESS": {"text": "1 Le Loi"}}}
        out = compact_result(full)
        self.assertEqual(out["fields"]["TIMESTAMP"], {"text": "Ngay 11/08/2020 16:02", "date": "11/08/2020",
                                                      "time": "16:02"})
        self.assertEqual(out["fields"]["TOTAL_COST"]["amount_vnd"], 74000)
        self.assertNotIn("SELLER", out["fields"])
        self.assertNotIn("items", out)


if __name__ == "__main__":
    unittest.main()
