import csv
import json
import os
import tempfile
import unittest

from src.common.bbox_utils import normalize_bbox
from src.features.layoutxlm_extractor import build_words_and_boxes
from src.preprocessing import token_dataset as td
from src.preprocessing.annotation_parser import AnnotationInstance, MCOCRRecord
from src.recognition.recognizer import OcrItem


def rect(x, y, w, h):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def inst(label, text, x, y, w, h, frame=(1000, 1000)):
    return AnnotationInstance(text=text, label=label, category_id=0, raw_bbox=[x, y, w, h],
                              normalized_bbox=[0, 0, 0, 0], segmentation=rect(x, y, w, h),
                              area=float(w * h), width=frame[0], height=frame[1])


def ocr_meta(items, orig=(1000, 1000), proc=(1000, 1000), angle=0.0):
    return {"orig_shape": [orig[1], orig[0]], "processed_shape": [proc[1], proc[0], 3], "skew_angle": angle,
            "ocr_items": [{"text": t, "bbox": b} for t, b in items]}


ITEMS = [("MINIMART ANAN", [100, 50, 500, 90]),
         ("Tong tien: 74,000", [100, 800, 500, 840]),
         ("Cam on", [100, 900, 300, 940])]
WORDS = ["MINIMART", "ANAN", "Tong", "tien:", "74,000", "Cam", "on"]
TAGS = ["B-SELLER", "I-SELLER", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST", "O", "O"]


def receipt(img_id="r1.jpg", frame=(1000, 1000), k=1.0):
    return MCOCRRecord(img_id, 2, 0.9, [
        inst("SELLER", "MINIMART ANAN", 90 * k, 45 * k, 420 * k, 50 * k, frame),
        inst("TOTAL_COST", "Tong tien: 74,000", 90 * k, 795 * k, 430 * k, 50 * k, frame)])


class TestLabels(unittest.TestCase):
    def test_label_set(self):
        self.assertEqual(len(td.LABELS), 9)
        self.assertEqual(td.LABELS[0], "O")
        self.assertIn("B-TOTAL_COST", td.LABELS)
        for lab in td.LABELS:
            self.assertEqual(td.ID2LABEL[td.LABEL2ID[lab]], lab)


class TestGeometry(unittest.TestCase):
    def test_polygon_points_shapes(self):
        self.assertEqual(td.polygon_points([0, 0, 10, 0, 10, 5]), [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)])
        self.assertEqual(len(td.polygon_points([[0, 0, 10, 0, 10, 5], [1, 1, 2, 2, 3, 3]])), 6)
        self.assertEqual(len(td.polygon_points([[0, 0], [10, 0], [10, 5]])), 3)
        self.assertEqual(td.polygon_points([]), [])

    def test_rotation_matrix_identity(self):
        m = td.rotation_matrix(0.0, (50, 50))
        for got, want in zip(m[0] + m[1], (1, 0, 0, 0, 1, 0)):
            self.assertAlmostEqual(got, want)

    def test_rotation_by_90_is_counter_clockwise_on_screen(self):
        g = td.ImageGeometry(100, 100, 100, 100, 90.0)
        x, y = g.map_points([(60, 50)])[0]
        self.assertAlmostEqual(x, 50.0, places=6)
        self.assertAlmostEqual(y, 40.0, places=6)

    def test_resize(self):
        g = td.ImageGeometry(200, 400, 100, 200)
        self.assertEqual(g.map_points([(100, 200)]), [(50.0, 100.0)])

    def test_annotation_frame_is_used_for_scaling(self):
        g = td.ImageGeometry(2000, 4000, 100, 200)
        self.assertEqual(g.map_points([(50, 100)], ref_size=(100, 200)), [(50.0, 100.0)])

    def test_bbox_is_clipped(self):
        g = td.ImageGeometry(100, 100, 100, 100)
        box = g.map_bbox([(-5, -5), (120, -5), (120, 50), (-5, 50)])
        self.assertEqual(box, [0.0, 0.0, 100.0, 50.0])

    def test_from_ocr_meta(self):
        g = td.ImageGeometry.from_ocr_meta(ocr_meta([], orig=(300, 400), proc=(150, 200), angle=-1.5))
        self.assertEqual((g.orig_w, g.orig_h, g.proc_w, g.proc_h, g.angle), (300, 400, 150, 200, -1.5))

    def test_frame_status(self):
        self.assertEqual(td.frame_status((768, 1024), (3024, 4032)), "ok")
        self.assertEqual(td.frame_status((1024, 768), (768, 1024)), "transposed")
        self.assertEqual(td.frame_status((1000, 1000), (768, 1024)), "aspect_mismatch")
        self.assertEqual(td.frame_status((0, 0), (10, 10)), "aspect_mismatch")


class TestWordsAndTags(unittest.TestCase):
    def test_words_equal_the_ones_used_at_inference(self):
        items = [OcrItem(0, 0, [], [10, 10, 200, 30], [10, 10, 200, 30], "Tong tien 74,000"),
                 OcrItem(1, 1, [], [10, 40, 100, 60], [10, 40, 100, 60], "Cam on")]
        words = td.items_to_words([{"text": i.text, "bbox": i.bbox} for i in items])
        want_words, want_boxes = build_words_and_boxes(items, 1000, 1000)
        self.assertEqual([w["text"] for w in words], want_words)
        self.assertEqual([normalize_bbox(w["bbox"], 1000, 1000, 1000, False) for w in words], want_boxes)

    def test_blank_items_are_skipped_and_source_index_kept(self):
        words = td.items_to_words([{"text": "   ", "bbox": [0, 0, 1, 1]}, {"text": "a", "bbox": [0, 0, 10, 10]}])
        self.assertEqual([(w["text"], w["item"]) for w in words], [("a", 1)])

    def test_assignment_threshold_and_tie_break(self):
        big = td.GtRegion("ADDRESS", "x", [0, 0, 1000, 1000])
        small = td.GtRegion("SELLER", "y", [0, 0, 100, 100])
        words = [{"text": "a", "bbox": [10, 10, 50, 30]},      # inside both: smaller region wins
                 {"text": "b", "bbox": [500, 500, 520, 510]}]  # only inside the big one
        fields, owners = td.assign_word_fields(words, [big, small])
        self.assertEqual(fields, ["SELLER", "ADDRESS"])
        self.assertEqual(owners, [1, 0])

        region = [td.GtRegion("SELLER", "y", [0, 0, 100, 10])]
        half = [{"text": "h", "bbox": [90, 0, 110, 10]}]       # exactly 50 % inside -> labelled
        less = [{"text": "l", "bbox": [92, 0, 112, 10]}]       # 40 % inside -> not labelled
        self.assertEqual(td.assign_word_fields(half, region)[0], ["SELLER"])
        self.assertEqual(td.assign_word_fields(less, region)[0], [None])

    def test_bio_tags(self):
        fields = [None, "SELLER", "SELLER", None, "ADDRESS", "SELLER"]
        self.assertEqual(td.bio_tags(fields), ["O", "B-SELLER", "I-SELLER", "O", "B-ADDRESS", "B-SELLER"])
        self.assertEqual(td.bio_tags(["TOTAL_COST", "TOTAL_COST", "ADDRESS"]),
                         ["B-TOTAL_COST", "I-TOTAL_COST", "B-ADDRESS"])


class TestBuildRecord(unittest.TestCase):
    def test_basic_record(self):
        record, diag = td.build_token_record("r1", ocr_meta(ITEMS), receipt())
        self.assertEqual(diag["status"], "ok")
        self.assertEqual(record["words"], WORDS)
        self.assertEqual(record["labels"], TAGS)
        self.assertEqual(len(record["boxes"]), len(WORDS))
        for box in record["boxes"]:
            self.assertTrue(all(0 <= v <= 1000 for v in box))
        self.assertEqual(record["boxes"][0][0], 100)
        self.assertEqual((record["width"], record["height"], record["image"]), (1000, 1000, "r1.jpg"))
        self.assertEqual(record["gt"], {"SELLER": ["MINIMART ANAN"], "TOTAL_COST": ["Tong tien: 74,000"]})
        self.assertEqual([r["covered"] for r in diag["regions"]], [True, True])
        self.assertEqual([r["cer"] for r in diag["regions"]], [0.0, 0.0])

    def test_annotation_frame_scaling(self):
        record, _ = td.build_token_record("r1", ocr_meta(ITEMS), receipt(frame=(500, 500), k=0.5))
        self.assertEqual(record["labels"], TAGS)

    def test_skew_is_applied_to_annotations(self):
        meta = ocr_meta([("xx", [0, 0, 10, 10]), ("74,000", [45, 30, 55, 40])],
                        orig=(100, 100), proc=(100, 100), angle=90.0)
        ann = MCOCRRecord("r.jpg", 1, 1.0, [inst("TOTAL_COST", "74,000", 60, 45, 10, 10, frame=(100, 100))])
        record, _ = td.build_token_record("r", meta, ann)
        self.assertEqual(record["labels"], ["O", "B-TOTAL_COST"])

    def test_frame_mismatch_skips_the_image(self):
        ann = receipt(frame=(768, 1024))
        record, diag = td.build_token_record("r1", ocr_meta(ITEMS), ann)
        self.assertIsNone(record)
        self.assertEqual(diag["status"], "frame_mismatch")
        self.assertEqual(diag["frame_detail"], "aspect_mismatch")

    def test_no_ocr(self):
        record, diag = td.build_token_record("r1", ocr_meta([]), receipt())
        self.assertIsNone(record)
        self.assertEqual(diag["status"], "no_ocr")


class TestSummary(unittest.TestCase):
    def test_coverage_and_status_counts(self):
        _, diag_a = td.build_token_record("a", ocr_meta(ITEMS), receipt("a.jpg"))
        far = MCOCRRecord("b.jpg", 1, 0.5, [inst("SELLER", "Shop", 500, 500, 100, 50)])
        _, diag_b = td.build_token_record("b", ocr_meta([("abc", [0, 0, 50, 20])]), far)
        _, diag_c = td.build_token_record("c", ocr_meta([]), receipt("c.jpg"))
        s = td.summarize_diagnostics([diag_a, diag_b, diag_c])
        self.assertEqual(s["images_by_status"], {"ok": 2, "no_ocr": 1})
        self.assertEqual(s["per_label"]["SELLER"]["regions"], 2)
        self.assertAlmostEqual(s["per_label"]["SELLER"]["coverage"], 0.5)
        self.assertAlmostEqual(s["per_label"]["SELLER"]["recovered"], 0.5)
        self.assertAlmostEqual(s["per_label"]["TOTAL_COST"]["coverage"], 1.0)
        self.assertEqual(s["words_per_image"]["max"], 7.0)
        self.assertAlmostEqual(s["words_per_image"]["mean"], 4.0)


class TestBuildDataset(unittest.TestCase):
    def _write_csv(self, path):
        def polys():
            return json.dumps([
                {"category_id": 15, "segmentation": rect(90, 45, 420, 50), "area": 21000,
                 "bbox": [90, 45, 420, 50], "width": 1000, "height": 1000},
                {"category_id": 18, "segmentation": rect(90, 795, 430, 50), "area": 21500,
                 "bbox": [90, 795, 430, 50], "width": 1000, "height": 1000}])
        cols = ["img_id", "anno_polygons", "anno_texts", "anno_labels", "anno_num", "anno_image_quality"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for img_id in ("r1.jpg", "r2.jpg", "r3.jpg"):
                w.writerow({"img_id": img_id, "anno_polygons": polys(),
                            "anno_texts": "MINIMART ANAN|||Tong tien: 74,000",
                            "anno_labels": "SELLER|||TOTAL_COST", "anno_num": 2, "anno_image_quality": 0.9})

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "ann.csv")
            ocr_dir = os.path.join(tmp, "ocr")
            out_dir = os.path.join(tmp, "out")
            manifest = os.path.join(tmp, "split.csv")
            os.makedirs(ocr_dir)
            self._write_csv(csv_path)
            for fid in ("r1", "r2"):
                with open(os.path.join(ocr_dir, fid + ".json"), "w", encoding="utf-8") as f:
                    json.dump(ocr_meta(ITEMS), f)
            with open(manifest, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["img_id", "split"])
                w.writeheader()
                for img_id, split in (("r1.jpg", "train"), ("r2.jpg", "validation"),
                                      ("r3.jpg", "train"), ("r4.jpg", "test")):
                    w.writerow({"img_id": img_id, "split": split})

            report = td.build_dataset(csv_path, ocr_dir, manifest, out_dir)

            self.assertEqual(report["records_written"], {"train": 1, "validation": 1})
            self.assertEqual(report["missing_ocr"], ["r3.jpg"])
            self.assertEqual(report["missing_annotation"], ["r4.jpg"])
            self.assertEqual(report["skipped"], [])
            self.assertEqual(report["by_split"]["train"]["per_label"]["SELLER"]["coverage"], 1.0)
            with open(os.path.join(out_dir, "train.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["labels"], TAGS)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "validation.jsonl")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "test.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "token_dataset_report.json")))


if __name__ == "__main__":
    unittest.main()
