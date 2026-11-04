import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from src.detection.detector import BaseTextDetector, TextRegion
from src.pipeline.runner import OutputDirs, ReceiptPipeline
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.recognition.recognizer import BaseTextRecognizer, TextRecognitionResult


class FakeDetector(BaseTextDetector):
    def __init__(self, boxes=None, fail=False):
        super().__init__()
        self.boxes, self.fail = boxes or [], fail

    def detect(self, image_bgr, scale_ratio=1.0):
        if self.fail:
            raise RuntimeError("boom")
        return [TextRegion([[b[0], b[1]], [b[2], b[3]]], b, b) for b in self.boxes]


class FakeRecognizer(BaseTextRecognizer):
    def recognize_batch(self, crops_rgb):
        return [TextRecognitionResult("text", 0.9) for _ in crops_rgb]


def make_images(folder, n):
    paths = []
    for i in range(n):
        p = os.path.join(folder, f"img{i}.jpg")
        cv2.imwrite(p, np.full((200, 300, 3), 255, np.uint8))
        paths.append(p)
    return paths


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = make_images(self.tmp.name, 5)
        self.dirs = OutputDirs.from_base(os.path.join(self.tmp.name, "out"))
        self.pre = ImagePreprocessor(enhance=False, max_skew=0.0)

    def pipeline(self, detector):
        return ReceiptPipeline(self.pre, detector, FakeRecognizer(), None, self.dirs)

    def test_process_writes_json_with_expected_fields(self):
        pl = self.pipeline(FakeDetector([[10, 10, 100, 40], [10, 60, 100, 90]]))
        stats, failed = pl.run(self.paths, progress=False)
        self.assertEqual(stats["ok"], 5)
        self.assertEqual(failed, [])
        meta = json.load(open(self.dirs.paths("img0")[0], encoding="utf-8"))
        self.assertEqual(meta["n_text_regions"], 2)
        self.assertEqual([it["region_id"] for it in meta["ocr_items"]], [0, 1])
        self.assertIn("skew_angle", meta)
        self.assertIn("total", meta["timing_sec"])

    def test_resume_skips_finished_images(self):
        pl = self.pipeline(FakeDetector([[10, 10, 100, 40]]))
        pl.run(self.paths, progress=False)
        stats, _ = pl.run(self.paths, progress=False)
        self.assertEqual(stats["skipped"], 5)
        self.assertEqual(stats["ok"], 0)

    def test_image_without_text_counts_as_done(self):
        pl = self.pipeline(FakeDetector([]))
        stats, _ = pl.run(self.paths, progress=False)
        self.assertEqual(stats["empty"], 5)
        self.assertTrue(pl.is_done("img0"))

    def test_early_abort_on_systematic_failure(self):
        pl = self.pipeline(FakeDetector(fail=True))
        stats, failed = pl.run(self.paths, abort_after=2, progress=False)
        self.assertEqual(stats["failed"], 2)          # stopped after 2, not 5
        self.assertIn("boom", failed[0][1])
        self.assertTrue(os.path.exists(os.path.join(self.dirs.base, "failed.json")))


if __name__ == "__main__":
    unittest.main()
