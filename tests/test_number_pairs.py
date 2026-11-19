import json
import os
import re
import tempfile
import unittest

import cv2
import numpy as np

from scripts.eval_numbers import digits, number_summary
from src.recognition.number_pairs import build_number_set, receipt_number_pairs, trusted_values
from src.recognition.synth_numbers import build_synth_set, number_text, render_number
from tests.test_extractor import make_meta

# a verified receipt (two items add up to the printed total); "8," + ",000" are two overlapping CRAFT boxes of 8,000
ITEMS = [("Ten hang SL Don gia Thanh tien", [100, 200, 900, 240]),
         ("Bim bim Oishi 48g", [100, 300, 500, 340]),
         ("2", [100, 350, 120, 390]), ("4,000", [200, 350, 300, 390]),
         ("8,", [400, 350, 432, 390]), (",000", [426, 350, 500, 390]),
         ("Sua Moc Chau 180ml", [100, 400, 500, 440]),
         ("1", [100, 450, 120, 490]), ("6,500", [200, 450, 300, 490]), ("6,500", [400, 450, 500, 490]),
         ("Tong cong: 14,500", [100, 800, 600, 840])]


class SynthNumbersTest(unittest.TestCase):
    def test_texts_look_like_receipt_numbers(self):
        rng = np.random.default_rng(0)
        texts = [number_text(rng) for _ in range(300)]
        self.assertTrue(all(re.fullmatch(r"[\d.,  ]+(đ| đ|\.00|,00)?", t) for t in texts))
        self.assertTrue(any(re.fullmatch(r"\d{1,3}([.,]\d{3})+", t) for t in texts))     # grouped amounts
        self.assertTrue(any(re.fullmatch(r"\d[.,]\d{2,3}", t) for t in texts))           # weights

    def test_render_and_build(self):
        img = render_number("439,850đ", None, np.random.default_rng(1))
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[1], img.shape[0])
        with tempfile.TemporaryDirectory() as tmp:
            rep = build_synth_set(tmp, "train", 5, seed=3, fonts=[])
            with open(os.path.join(tmp, "train", "pairs.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual((rep["written"], len(rows), rows[0]["kind"]), (5, 5, "synth"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "train", rows[0]["crop"])))
            rep = build_synth_set(tmp, "validation", 20, seed=3, fonts=[], vocab_chars="0123456789")
            self.assertEqual(rep["written"] + rep["out_of_vocab"], 20)


class NumberPairsTest(unittest.TestCase):
    def test_trusted_values_only_for_verified_receipts(self):
        meta = make_meta(ITEMS)
        from src.pipeline.extractor import record_from_ocr
        rec = record_from_ocr(meta, "x.jpg")
        self.assertEqual(trusted_values(rec["words"], rec["boxes"]), {4000, 8000, 6500, 14500})
        bad = make_meta(ITEMS[:-1] + [("Tong cong: 99,000", [100, 800, 600, 840])])
        rec = record_from_ocr(bad, "x.jpg")
        self.assertEqual(trusted_values(rec["words"], rec["boxes"]), set())
        self.assertEqual(receipt_number_pairs(bad), [])

    def test_pairs_of_a_verified_receipt(self):
        pairs = receipt_number_pairs(make_meta(ITEMS))
        split = [p for p in pairs if p["kind"] == "number_split"]
        self.assertEqual([(p["text"], p["bbox"], p["ocr"]) for p in split], [("8,000", [400, 350, 500, 390], "8, ,000")])
        self.assertEqual(sorted(p["text"] for p in pairs if p["kind"] == "number"), ["14,500", "4,000", "6,500", "6,500"])

    def test_build_writes_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            ocr, img = os.path.join(tmp, "ocr"), os.path.join(tmp, "img")
            os.makedirs(ocr), os.makedirs(img)
            with open(os.path.join(ocr, "r1.json"), "w", encoding="utf-8") as f:
                json.dump(make_meta(ITEMS), f)
            cv2.imwrite(os.path.join(img, "r1.jpg"), np.full((1000, 1000, 3), 255, np.uint8))
            manifest = os.path.join(tmp, "m.csv")
            with open(manifest, "w", encoding="utf-8") as f:
                f.write("img_id,split\nr1.jpg,train\n")
            rep = build_number_set(ocr, img, manifest, ["train"], os.path.join(tmp, "out"))
            self.assertEqual((rep["train"]["verified_receipts"], rep["train"]["number_split"], rep["train"]["number"]),
                             (1, 1, 4))


class EvalNumbersTest(unittest.TestCase):
    def test_digit_metric_ignores_separators(self):
        self.assertEqual(digits("45.000đ"), "45000")
        s = number_summary(["45,000", "12,000", "0.57"], ["45.000", "2,000", "0.57"])
        self.assertAlmostEqual(s["digits"], 2 / 3)
        self.assertAlmostEqual(s["exact"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
