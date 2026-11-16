import os
import tempfile
import unittest

import cv2
import numpy as np

from src.evaluation import ocr_ablation as oa
from src.preprocessing.annotation_parser import AnnotationInstance

TEXTS = ("Cua hang ABC", "Tong tien: 74,000")


def rect(x, y, w, h):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def inst(label, text, x, y, w, h, frame=(300, 200)):
    return AnnotationInstance(text=text, label=label, category_id=0, raw_bbox=[x, y, w, h],
                              normalized_bbox=[0, 0, 0, 0], segmentation=rect(x, y, w, h),
                              area=float(w * h), width=frame[0], height=frame[1])


class Res:
    def __init__(self, text):
        self.text = text


class ConstRecognizer:
    def __init__(self, text):
        self.text = text

    def recognize_batch(self, crops):
        return [Res(self.text) for _ in crops]


class QueueRecognizer:
    def __init__(self, texts):
        self.texts = list(texts)

    def recognize_batch(self, crops):
        return [Res(self.texts.pop(0)) for _ in crops]


def make_items(folder, n=3, frame=(300, 200), qualities=(0.2, 0.5, 0.9)):
    items = []
    for i in range(n):
        path = os.path.join(folder, f"r{i}.jpg")
        cv2.imwrite(path, np.full((200, 300, 3), 255, np.uint8))
        items.append({"img_id": f"r{i}.jpg", "path": path, "quality": qualities[i],
                      "instances": [inst("SELLER", TEXTS[0], 20, 20, 100, 30, frame),
                                    inst("TOTAL_COST", TEXTS[1], 20, 100, 150, 30, frame)]})
    return items


class TestDegradations(unittest.TestCase):
    def test_none_is_a_copy(self):
        img = np.full((10, 10, 3), 50, np.uint8)
        out = oa.degrade(img, "none")
        self.assertTrue((out == img).all())
        self.assertIsNot(out, img)

    def test_dark(self):
        out = oa.degrade(np.full((10, 10, 3), 200, np.uint8), "dark")
        self.assertTrue((out == 70).all())

    def test_blur_removes_high_frequencies(self):
        img = np.zeros((40, 40, 3), np.uint8)
        img[:, ::2] = 255
        self.assertLess(oa.degrade(img, "blur").std(), img.std())

    def test_noise_is_seeded(self):
        img = np.full((20, 20, 3), 128, np.uint8)
        a, b, c = oa.degrade(img, "noise", 1), oa.degrade(img, "noise", 1), oa.degrade(img, "noise", 2)
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())
        self.assertFalse((a == img).all())
        self.assertEqual(a.dtype, np.uint8)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            oa.degrade(np.zeros((4, 4, 3), np.uint8), "fog")


class TestHelpers(unittest.TestCase):
    def test_quality_groups_are_thirds(self):
        groups = oa.quality_groups({f"r{i}": q for i, q in enumerate([0.9, 0.1, 0.5, 0.3, 0.7, 0.2])})
        counts = {g: list(groups.values()).count(g) for g in oa.QUALITY_GROUPS}
        self.assertEqual(counts, {"low": 2, "mid": 2, "high": 2})
        self.assertEqual(groups["r1"], "low")
        self.assertEqual(groups["r0"], "high")

    def test_crop_regions(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[:, :] = (255, 0, 0)                                   # BGR blue
        crops, kept = oa.crop_regions(img, [[10, 10, 50, 30], [150, 150, 160, 160], [90, 90, 200, 200]])
        self.assertEqual(kept, [0, 2])
        self.assertEqual(crops[0].shape, (24, 46, 3))
        self.assertEqual(crops[1].shape, (12, 13, 3))
        self.assertEqual(crops[0][0, 0].tolist(), [0, 0, 255])    # returned as RGB

    def test_paired_difference(self):
        base = {f"r{i}": (10, 100) for i in range(20)}
        same = oa.paired_cer_difference(base, dict(base))
        self.assertEqual((same["diff"], same["low"], same["high"]), (0.0, 0.0, 0.0))
        worse = oa.paired_cer_difference(base, {k: (20, 100) for k in base})
        self.assertAlmostEqual(worse["diff"], 0.1)
        self.assertGreater(worse["low"], 0.0)
        self.assertEqual(worse["share_better"], 0.0)
        self.assertEqual(oa.paired_cer_difference({}, base)["n"], 0)


class TestRunAblation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_perfect_reading_gives_zero_error_for_every_configuration(self):
        items = make_items(self.tmp.name)
        refs = [i.text for it in items for i in it["instances"]]
        result = oa.run_ablation(items, QueueRecognizer(refs * len(oa.CONFIGS)))
        self.assertEqual(len(result["rows"]), 3 * 2 * len(oa.CONFIGS))
        summary = oa.summarize(result["rows"])
        self.assertEqual(set(summary), {f"none/{c}" for c in oa.CONFIGS})
        for s in summary.values():
            self.assertEqual((s["cer"], s["wer"], s["regions"]), (0.0, 0.0, 6))
        self.assertEqual(result["skipped"], {"unreadable": 0, "frame_mismatch": 0})

    def test_empty_reading_gives_full_error(self):
        items = make_items(self.tmp.name)
        summary = oa.summarize(oa.run_ablation(items, ConstRecognizer(""))["rows"])
        for s in summary.values():
            self.assertEqual((s["cer"], s["wer"]), (1.0, 1.0))
            self.assertEqual(s["by_group"]["low"]["regions"], 2)
            self.assertEqual(set(s["by_label"]), {"SELLER", "TOTAL_COST"})

    def test_unreadable_and_mismatching_receipts_are_skipped_and_counted(self):
        good = make_items(self.tmp.name, n=1)
        bad_frame = make_items(self.tmp.name, n=2, frame=(500, 500))[1:]
        missing = [{"img_id": "gone.jpg", "path": os.path.join(self.tmp.name, "gone.jpg"), "quality": 0.4,
                    "instances": good[0]["instances"]}]
        result = oa.run_ablation(good + bad_frame + missing, ConstRecognizer("x"))
        self.assertEqual(result["skipped"], {"unreadable": 1, "frame_mismatch": 1})
        self.assertEqual({r["img_id"] for r in result["rows"]}, {"r0.jpg"})

    def test_degradations_multiply_the_rows(self):
        items = make_items(self.tmp.name, n=1)
        result = oa.run_ablation(items, ConstRecognizer("x"), degradations=("none", "dark"))
        self.assertEqual({r["degradation"] for r in result["rows"]}, {"none", "dark"})
        self.assertEqual(len(result["rows"]), 2 * 2 * len(oa.CONFIGS))

    def test_report_and_markdown(self):
        items = make_items(self.tmp.name)
        result = oa.run_ablation(items, ConstRecognizer("x"))
        report = oa.build_report(result, ["none"], list(oa.CONFIGS), n_boot=50)
        self.assertIn("none/deskew vs none/none", report["paired_vs_none"])
        self.assertEqual(report["n_receipts"], 3)
        text = oa.format_markdown(report)
        self.assertIn("| none/full |", text)
        self.assertIn("Paired difference", text)


if __name__ == "__main__":
    unittest.main()
