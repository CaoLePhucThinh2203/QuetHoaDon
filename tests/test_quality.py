import tempfile
import unittest

from src.pipeline.extractor import ReceiptExtractor, compact_result
from src.pipeline.demo_view import items_summary
from src.recognition.quality import readability, repeated_share
from tests.test_extractor import make_meta, onehot, write_image
from tests.test_extractor_items import ITEMS

# what the fine-tuned VietOCR writes on an unreadable photo (train ...013crsgq): learned phrases, again and again, whose
# invented amounts even add up ("10.000đ" items, "Tổng thanh toán 10.000đ")
INVENTED = [("5 s 8 S 10.000đ", [100, 50, 500, 90])] + [
    (text, [100, 100 + 40 * i, 600, 130 + 40 * i]) for i, text in enumerate(
        ["Tổng thanh toán", "104 Trần Phú - Quảng Ninh", "Tên hàng SL Thành tiền", "Phí 1 10.000đ 10.000đ",
         "104 Trần Phú - Quảng Ninh", "Tổng thanh toán", "104 Trần Phú - Quảng Ninh", "Tổng thanh toán 10.000đ",
         "104 Trần Phú - Quảng Ninh", "Tổng thanh toán"])]


class ReadabilityTest(unittest.TestCase):
    def test_repeated_share(self):
        self.assertEqual(repeated_share("a b c a b c a b c".split()), 1.0)
        self.assertEqual(repeated_share("a b c d e f g h".split()), 0.0)
        self.assertEqual(repeated_share(["a", "b"]), 0.0)

    def test_normal_receipt_is_readable_and_invented_text_is_not(self):
        normal = " ".join(t for t, _ in ITEMS).split()
        self.assertFalse(readability(normal)["unreadable"])
        invented = " ".join(t for t, _ in INVENTED).split()
        q = readability(invented)
        self.assertTrue(q["unreadable"])
        self.assertGreaterEqual(q["repeated_share"], 0.3)
        self.assertFalse(readability(invented[:10])["unreadable"])       # too few words to judge


class ExtractorQualityTest(unittest.TestCase):
    def run_extractor(self, ocr_items, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            return ReceiptExtractor(lambda rec, d: onehot(["O"] * len(rec["words"])), **kw).extract_from_ocr(
                make_meta(ocr_items), write_image(tmp))

    def test_off_by_default(self):
        result = self.run_extractor(INVENTED, items=True)
        self.assertNotIn("warning", result)
        self.assertNotIn("quality", result)

    def test_invented_text_gives_a_warning_and_no_numbers(self):
        result = self.run_extractor(INVENTED, items=True, total_from_items=True, quality_check=True)
        self.assertEqual(result["warning"], "unreadable")
        self.assertEqual(result["items_check"]["status"], "unreadable")        # never a false "verified"
        self.assertNotEqual(result["fields"]["TOTAL_COST"].get("source"), "items")
        compact = compact_result(result)
        self.assertEqual(compact, {"file_id": result["file_id"], "fields": {}, "warning": "unreadable"})
        self.assertIn("khó đọc", items_summary(compact))

    def test_normal_receipt_unchanged(self):
        plain = self.run_extractor(ITEMS, items=True)
        checked = self.run_extractor(ITEMS, items=True, quality_check=True)
        self.assertNotIn("warning", checked)
        self.assertFalse(checked["quality"]["unreadable"])
        self.assertEqual(compact_result(plain), compact_result(checked))


if __name__ == "__main__":
    unittest.main()
