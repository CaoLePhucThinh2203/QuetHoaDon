import unittest

from src.evaluation.ocr_errors import OCR_ERROR_TYPES, classify_ocr_error, count_types


class TestClassify(unittest.TestCase):
    def test_types(self):
        cases = [
            ("Trần Phú", "tran  phu", "diacritics"),
            ("Trần Phú", "Trần Phú ", "none"),
            ("Trần Phú", "TrầnPhú", "spacing"),
            ("ĐT: 0123456", "ĐT: 0123465", "digits"),
            ("Cửa hàng bách hóa Hoàng Long", "Hoàng Long", "truncated"),
            ("ABC", "ABC 123 xyz mno", "extra_text"),
            ("nguyen van a", "nguyen van b", "other"),
        ]
        for gold, text, expected in cases:
            self.assertEqual(classify_ocr_error(gold, text), expected, (gold, text))

    def test_missing_text_counts_as_truncated(self):
        self.assertEqual(classify_ocr_error("Cua hang ABC", None), "truncated")

    def test_count_types_joins_multi_region_gold(self):
        rows = [{"field": "ADDRESS", "gold": "12 Tran Phu | Ha Noi", "oracle": "12 Tran Phu Ha Noi"},
                {"field": "ADDRESS", "gold": "Ha Noi", "oracle": "Ha"},
                {"field": "SELLER", "gold": "ABC", "oracle": "ABC 123 xyz mno"}]
        c = count_types(rows)
        self.assertEqual(c["ADDRESS"]["none"], 1)
        self.assertEqual(c["ADDRESS"]["truncated"], 1)
        self.assertEqual(c["SELLER"]["extra_text"], 1)
        self.assertEqual(set(OCR_ERROR_TYPES) | {"none"}, set(c["SELLER"]))


if __name__ == "__main__":
    unittest.main()
