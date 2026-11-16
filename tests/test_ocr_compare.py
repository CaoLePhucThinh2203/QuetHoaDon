import unittest

from src.detection.postprocess import filter_nested_items
from src.evaluation.ocr_compare import field_oracle, field_words, format_oracle_table, word_identity


def rec(img_id, words, labels, gt):
    return {"img_id": img_id, "words": words, "labels": labels, "gt": gt}


class TestFieldOracle(unittest.TestCase):
    def test_exact_and_cer(self):
        r1 = rec("a", ["Địa", "chỉ:", "12", "Trần", "Phú", "x"],
                 ["O", "O", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS", "O"], {"ADDRESS": ["12 Trần Phú"]})
        r2 = rec("b", ["12", "Tran", "Phu"], ["B-ADDRESS", "I-ADDRESS", "I-ADDRESS"], {"ADDRESS": ["12 Trần Phú"]})
        r3 = rec("c", ["abc"], ["O"], {"SELLER": ["ABC"]})
        o = field_oracle([r1, r2, r3])
        self.assertEqual((o["ADDRESS"]["n"], o["ADDRESS"]["exact"]), (2, 1))
        self.assertGreater(o["ADDRESS"]["mean_cer"], 0)
        self.assertEqual((o["SELLER"]["n"], o["SELLER"]["covered"], o["SELLER"]["exact"]), (1, 0, 0))
        self.assertEqual(o["TOTAL_COST"]["n"], 0)
        self.assertIn("| x |", format_oracle_table({"x": o}))

    def test_field_words(self):
        r = rec("a", ["A", "B", "C"], ["B-SELLER", "O", "I-SELLER"], {})
        self.assertEqual(field_words(r, "SELLER"), "A C")


class TestIdentity(unittest.TestCase):
    def test_counts(self):
        a = [{"img_id": "1", "words": ["x"]}, {"img_id": "2", "words": ["y"]}, {"img_id": "3", "words": []}]
        b = [{"img_id": "1", "words": ["x"]}, {"img_id": "2", "words": ["z"]}]
        i = word_identity(a, b)
        self.assertEqual((i["common"], i["identical"], i["only_in_a"], i["only_in_b"]), (2, 1, 1, 0))
        self.assertEqual(i["first_differences"], ["2"])


class TestFilterNestedItems(unittest.TestCase):
    def test_threshold_and_renumbering(self):
        items = [{"text": "big", "bbox": [0, 0, 100, 20], "region_id": 0, "reading_order": 0},
                 {"text": "half", "bbox": [90, 0, 110, 20], "region_id": 1, "reading_order": 1},
                 {"text": "far", "bbox": [300, 0, 340, 20], "region_id": 2, "reading_order": 2}]
        self.assertEqual([i["text"] for i in filter_nested_items(items, 0.8)], ["big", "half", "far"])
        kept = filter_nested_items(items, 0.5)
        self.assertEqual([i["text"] for i in kept], ["big", "far"])
        self.assertEqual([i["region_id"] for i in kept], [0, 1])
        self.assertEqual(items[2]["region_id"], 2)                 # input not modified


if __name__ == "__main__":
    unittest.main()
