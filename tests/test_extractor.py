import os
import tempfile
import unittest

import cv2
import numpy as np

from src.pipeline.extractor import ReceiptExtractor, empty_fields, fields_from_probs, record_from_ocr
from src.preprocessing.token_dataset import LABEL2ID, LABELS

ITEMS = [("MINIMART ANAN", [100, 50, 500, 90]),
         ("Ngay 11/08/2020", [100, 100, 500, 140]),
         ("Tong tien: 74,000", [100, 800, 500, 840]),
         ("Cam on", [100, 900, 300, 940])]
WORDS = ["MINIMART", "ANAN", "Ngay", "11/08/2020", "Tong", "tien:", "74,000", "Cam", "on"]
TAGS = ["B-SELLER", "I-SELLER", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST", "O", "O"]


def make_meta(items=ITEMS):
    return {"file_id": "r1", "orig_shape": [1000, 1000], "processed_shape": [1000, 1000, 3], "skew_angle": 0.0,
            "ocr_items": [{"text": t, "bbox": b} for t, b in items],
            "timing_sec": {"preprocess": 0.1, "detect": 0.2, "ocr": 0.3, "features": 0.0, "total": 0.6}}


def onehot(tags):
    probs = np.full((len(tags), len(LABELS)), 0.01 / (len(LABELS) - 1))
    for i, t in enumerate(tags):
        probs[i, LABEL2ID[t]] = 0.99
    return probs


def write_image(folder, name="r1.jpg"):
    path = os.path.join(folder, name)
    cv2.imwrite(path, np.full((100, 100, 3), 255, np.uint8))
    return path


class FakeOcrPipeline:
    def __init__(self, meta):
        self.meta = meta

    def infer(self, path):
        return self.meta, np.full((100, 100, 3), 255, np.uint8)


class TestRecordAndFields(unittest.TestCase):
    def test_record_from_ocr(self):
        rec = record_from_ocr(make_meta(), "r1.jpg")
        self.assertEqual(rec["words"], WORDS)
        self.assertEqual(len(rec["boxes"]), len(WORDS))
        self.assertEqual(len(rec["pixel_boxes"]), len(WORDS))
        self.assertEqual(set(rec["labels"]), {"O"})
        self.assertEqual((rec["img_id"], rec["image"]), ("r1", "r1.jpg"))
        for box in rec["boxes"]:
            self.assertTrue(all(0 <= v <= 1000 for v in box))

    def test_fields_from_probs(self):
        rec = record_from_ocr(make_meta(), "r1.jpg")
        fields, tags = fields_from_probs(rec, onehot(TAGS))
        self.assertEqual(tags, TAGS)
        self.assertEqual(fields["SELLER"]["text"], "MINIMART ANAN")
        self.assertEqual(fields["SELLER"]["status"], "found")
        self.assertEqual(fields["TOTAL_COST"]["amount_vnd"], 74000)
        self.assertEqual(fields["TOTAL_COST"]["source"], "model")
        self.assertEqual(fields["TIMESTAMP"]["date"], "11/08/2020")
        self.assertEqual(fields["ADDRESS"]["status"], "not_found")
        self.assertIsNone(fields["ADDRESS"]["text"])

    def test_rule_fills_a_total_the_tagger_missed(self):
        words_meta = make_meta([("Tong cong 74,000", [100, 800, 500, 840])])
        rec = record_from_ocr(words_meta, "r1.jpg")
        fields, _ = fields_from_probs(rec, onehot(["O"] * len(rec["words"])))
        self.assertEqual(fields["TOTAL_COST"]["amount_vnd"], 74000)
        self.assertEqual(fields["TOTAL_COST"]["source"], "rule")

    def test_empty_fields(self):
        f = empty_fields()
        self.assertEqual({v["status"] for v in f.values()}, {"not_found"})
        self.assertIsNone(f["TOTAL_COST"]["amount_vnd"])


class TestExtractor(unittest.TestCase):
    def test_extract_from_ocr(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = write_image(tmp)
            calls = []

            def probs_fn(record, image_dir):
                calls.append((record, image_dir))
                return onehot(TAGS)

            result = ReceiptExtractor(probs_fn).extract_from_ocr(make_meta(), image)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], os.path.abspath(tmp))
        self.assertEqual(calls[0][0]["image"], "r1.jpg")
        self.assertEqual(result["fields"]["SELLER"]["text"], "MINIMART ANAN")
        self.assertEqual(result["n_words"], len(WORDS))
        self.assertEqual([w["tag"] for w in result["words"]], TAGS)
        self.assertEqual(result["bbox_unit"], "pixel")
        self.assertEqual(result["timing_sec"]["preprocess"], 0.1)
        self.assertIn("layoutxlm", result["timing_sec"])
        self.assertGreaterEqual(result["timing_sec"]["total"], 0.6)

    def test_no_words_skips_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = write_image(tmp)

            def probs_fn(record, image_dir):
                raise AssertionError("the model must not run without words")

            result = ReceiptExtractor(probs_fn).extract_from_ocr(make_meta([]), image)
        self.assertEqual({f["status"] for f in result["fields"].values()}, {"not_found"})
        self.assertEqual(result["n_words"], 0)

    def test_extract_runs_ocr_and_hands_the_preprocessed_image_to_the_model(self):
        seen = {}

        def probs_fn(record, image_dir):
            seen["exists"] = os.path.exists(os.path.join(image_dir, record["image"]))
            seen["image"] = record["image"]
            return onehot(TAGS)

        extractor = ReceiptExtractor(probs_fn, FakeOcrPipeline(make_meta()))
        result = extractor.extract("/some/photo.jpg")
        self.assertTrue(seen["exists"])
        self.assertEqual(seen["image"], "r1.jpg")
        self.assertEqual(result["image"], "/some/photo.jpg")
        self.assertEqual(result["fields"]["TOTAL_COST"]["amount_vnd"], 74000)

    def test_extract_needs_an_ocr_pipeline(self):
        with self.assertRaises(RuntimeError):
            ReceiptExtractor(lambda r, d: onehot([])).extract("x.jpg")

    def test_extract_from_record_ignores_gold_labels(self):
        seen = {}

        def probs_fn(record, image_dir):
            seen["labels"] = set(record["labels"])
            seen["dir"] = image_dir
            return onehot(TAGS)

        record = {"img_id": "a", "file_id": "a", "image": "a.jpg", "words": WORDS,
                  "boxes": [[0, 0, 10, 10]] * len(WORDS), "labels": TAGS, "gt": {"SELLER": ["x"]}}
        result = ReceiptExtractor(probs_fn).extract_from_record(record, "/images")
        self.assertEqual(seen["labels"], {"O"})
        self.assertEqual(seen["dir"], "/images")
        self.assertEqual(result["bbox_unit"], "normalized_1000")
        self.assertEqual(result["fields"]["TOTAL_COST"]["amount_vnd"], 74000)


if __name__ == "__main__":
    unittest.main()
