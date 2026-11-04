import unittest

from src.features.layoutxlm_extractor import build_words_and_boxes, split_box_by_tokens
from src.recognition.recognizer import OcrItem


def item(text, bbox):
    return OcrItem(0, 0, [], bbox, bbox, text)


class TestSplitBox(unittest.TestCase):
    def test_words_stay_inside_and_in_order(self):
        parts = split_box_by_tokens([100, 10, 400, 30], ["Tong", "tien", "thanh", "toan"])
        self.assertEqual(parts[0][0], 100)
        self.assertLessEqual(parts[-1][2], 400)
        for a, b in zip(parts, parts[1:]):
            self.assertLessEqual(a[2], b[0] + 1)

    def test_single_token_keeps_box(self):
        self.assertEqual(split_box_by_tokens([10, 5, 110, 25], ["abc"]), [[10, 5, 110, 25]])


class TestBuildWords(unittest.TestCase):
    def test_normalised_to_1000(self):
        words, boxes = build_words_and_boxes([item("27,000", [0, 0, 500, 100])], 1000, 1000)
        self.assertEqual(words, ["27,000"])
        self.assertEqual(boxes, [[0, 0, 500, 100]])

    def test_max_words_and_blank_items(self):
        items = [item("a b c", [0, 0, 90, 10]), item("   ", [0, 0, 10, 10])]
        words, boxes = build_words_and_boxes(items, 100, 100, max_words=2)
        self.assertEqual(words, ["a", "b"])
        self.assertEqual(len(boxes), 2)
        for b in boxes:
            self.assertTrue(all(0 <= v <= 1000 for v in b))


if __name__ == "__main__":
    unittest.main()
