"""Unit tests for reproducible MC-OCR dataset preparation module."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.preprocessing.dataset_preparer import (
    DataLeakageError,
    ManifestRecord,
    check_data_leakage,
    check_manifest_leakage,
    prepare_dataset,
    prepare_mcocr_dataset,
    split_invoices,
)


class TestDatasetPreparer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.image_dir = self.base_path / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.base_path / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.base_path / "train_annotations.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_image(self, filename: str):
        """Create a placeholder dummy image file."""
        img_file = self.image_dir / filename
        img_file.write_bytes(b"dummy image data")
        return img_file

    def _write_csv(self, rows):
        """Write rows into CSV file."""
        fieldnames = [
            "img_id",
            "anno_num",
            "anno_image_quality",
            "anno_texts",
            "anno_labels",
            "anno_polygons",
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def test_split_invoices_reproduces_mcocr_experiment_counts(self):
        """Reproduce Kaggle MC-OCR experiment with 1153 valid invoices:

        train: 922, validation: 115, test: 116.
        """
        mock_invoices = [f"mcocr_train_{i:06d}.jpg" for i in range(1153)]
        splits = split_invoices(mock_invoices, random_seed=42)

        self.assertEqual(len(splits["train"]), 922)
        self.assertEqual(len(splits["validation"]), 115)
        self.assertEqual(len(splits["test"]), 116)

        # Ensure no overlap
        check_data_leakage(splits, expected_img_ids=mock_invoices)

    def test_prepare_dataset_end_to_end(self):
        # Create 10 dummy images
        for i in range(1, 11):
            self._create_dummy_image(f"receipt_{i:02d}.jpg")

        rows = []
        # 10 valid invoices with images
        for i in range(1, 11):
            rows.append({
                "img_id": f"receipt_{i:02d}.jpg",
                "anno_num": "2",
                "anno_image_quality": "0.95",
                "anno_texts": f"Cua Hang {i}|||{i * 10000} VND",
                "anno_labels": "SELLER|||TOTAL_COST",
                "anno_polygons": (
                    f"[{{'category_id': 15, 'bbox': [10, 20, 100, 30], 'width': 1000, 'height': 1000, 'segmentation': []}},"
                    f" {{'category_id': 18, 'bbox': [10, 200, 80, 25], 'width': 1000, 'height': 1000, 'segmentation': []}}]"
                ),
            })

        # 1 valid annotation whose image file is MISSING
        rows.append({
            "img_id": "receipt_missing.jpg",
            "anno_num": "1",
            "anno_image_quality": "0.80",
            "anno_texts": "Missing Image Store",
            "anno_labels": "SELLER",
            "anno_polygons": "[{'category_id': 15, 'bbox': [10, 10, 50, 20], 'width': 1000, 'height': 1000, 'segmentation': []}]",
        })

        # 1 empty row (anno_num == 0, image exists)
        self._create_dummy_image("receipt_empty.jpg")
        rows.append({
            "img_id": "receipt_empty.jpg",
            "anno_num": "0",
            "anno_image_quality": "0.50",
            "anno_texts": "",
            "anno_labels": "",
            "anno_polygons": "[]",
        })

        # 1 invalid row (length mismatch, image exists)
        self._create_dummy_image("receipt_invalid.jpg")
        rows.append({
            "img_id": "receipt_invalid.jpg",
            "anno_num": "2",
            "anno_image_quality": "0.50",
            "anno_texts": "Only One",
            "anno_labels": "SELLER|||TOTAL_COST",
            "anno_polygons": "[]",
        })

        self._write_csv(rows)

        summary = prepare_dataset(
            csv_path=self.csv_path,
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            random_seed=42,
        )

        # 1. Verify output files exist
        clean_manifest_path = self.output_dir / "mcocr_clean_manifest.csv"
        split_manifest_path = self.output_dir / "mcocr_manifest_split.csv"
        summary_json_path = self.output_dir / "dataset_summary.json"

        self.assertTrue(clean_manifest_path.exists())
        self.assertTrue(split_manifest_path.exists())
        self.assertTrue(summary_json_path.exists())

        # 2. Verify summary dictionary metrics
        self.assertEqual(summary["total_original_records"], 13)
        self.assertEqual(summary["valid_invoices_used"], 10)
        self.assertEqual(summary["excluded_empty_records"], 1)
        self.assertEqual(summary["invalid_records"], 1)
        self.assertEqual(summary["missing_image_records"], 1)
        self.assertEqual(summary["annotated_regions"], 20)  # 10 invoices * 2 regions each
        self.assertEqual(summary["label_distribution"], {"SELLER": 10, "TOTAL_COST": 10})

        # Check split invoice counts (train: 8, validation: 1, test: 1)
        self.assertEqual(summary["invoice_counts_by_split"]["train"], 8)
        self.assertEqual(summary["invoice_counts_by_split"]["validation"], 1)
        self.assertEqual(summary["invoice_counts_by_split"]["test"], 1)
        self.assertNotIn("val", summary["invoice_counts_by_split"])

        # Check region counts by split (8*2, 1*2, 1*2)
        self.assertEqual(summary["region_counts_by_split"]["train"], 16)
        self.assertEqual(summary["region_counts_by_split"]["validation"], 2)
        self.assertEqual(summary["region_counts_by_split"]["test"], 2)
        self.assertNotIn("val", summary["region_counts_by_split"])

        # Check leakage check result is plain string "PASSED"
        self.assertEqual(summary["leakage_check_result"], "PASSED")
        self.assertIsInstance(summary["leakage_check_result"], str)

        # 3. Verify exported JSON matches
        with open(summary_json_path, "r", encoding="utf-8") as f:
            saved_summary = json.load(f)
        self.assertEqual(saved_summary["total_original_records"], 13)
        self.assertEqual(saved_summary["valid_invoices_used"], 10)
        self.assertEqual(saved_summary["excluded_empty_records"], 1)
        self.assertEqual(saved_summary["invalid_records"], 1)
        self.assertEqual(saved_summary["missing_image_records"], 1)
        self.assertEqual(saved_summary["annotated_regions"], 20)
        self.assertEqual(saved_summary["invoice_counts_by_split"]["train"], 8)
        self.assertEqual(saved_summary["invoice_counts_by_split"]["validation"], 1)
        self.assertEqual(saved_summary["invoice_counts_by_split"]["test"], 1)
        self.assertEqual(saved_summary["leakage_check_result"], "PASSED")

        # 4. Verify mcocr_clean_manifest.csv
        with open(clean_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            clean_rows = list(reader)

        self.assertEqual(len(clean_rows), 20)
        for row in clean_rows:
            self.assertIn("img_id", row)
            self.assertIn("image_path", row)
            self.assertIn("text", row)
            self.assertIn("label", row)
            self.assertIn("raw_bbox", row)
            self.assertIn("normalized_bbox", row)
            self.assertNotIn("split", row)
            self.assertNotIn("receipt_missing.jpg", row["img_id"])
            self.assertNotIn("receipt_empty.jpg", row["img_id"])
            self.assertNotIn("receipt_invalid.jpg", row["img_id"])

        # 5. Verify mcocr_manifest_split.csv
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            split_rows = list(reader)

        self.assertEqual(len(split_rows), 20)
        for row in split_rows:
            self.assertIn("split", row)
            self.assertIn(row["split"], ("train", "validation", "test"))

    def test_invoice_level_splitting_integrity(self):
        """Ensure all regions of a single receipt are in the same split (never split annotations)."""
        for i in range(1, 11):
            self._create_dummy_image(f"receipt_{i:02d}.jpg")

        rows = []
        for i in range(1, 11):
            rows.append({
                "img_id": f"receipt_{i:02d}.jpg",
                "anno_num": "4",
                "anno_image_quality": "0.90",
                "anno_texts": "Seller A|||Address B|||2021-01-01|||100000",
                "anno_labels": "SELLER|||ADDRESS|||TIMESTAMP|||TOTAL_COST",
                "anno_polygons": (
                    "[{'category_id': 15, 'bbox': [10, 10, 100, 20], 'width': 1000, 'height': 1000, 'segmentation': []},"
                    " {'category_id': 16, 'bbox': [10, 40, 100, 20], 'width': 1000, 'height': 1000, 'segmentation': []},"
                    " {'category_id': 17, 'bbox': [10, 70, 100, 20], 'width': 1000, 'height': 1000, 'segmentation': []},"
                    " {'category_id': 18, 'bbox': [10, 100, 100, 20], 'width': 1000, 'height': 1000, 'segmentation': []}]"
                ),
            })
        self._write_csv(rows)

        prepare_dataset(
            csv_path=self.csv_path,
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            random_seed=42,
        )

        split_manifest_path = self.output_dir / "mcocr_manifest_split.csv"
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            split_rows = list(reader)

        # Group split by img_id
        img_splits = {}
        for row in split_rows:
            img_splits.setdefault(row["img_id"], set()).add(row["split"])

        # Every image must have exactly 1 assigned split across all its annotations
        for img_id, splits in img_splits.items():
            self.assertEqual(
                len(splits),
                1,
                f"Image {img_id} was split into multiple splits: {splits}",
            )

    def test_reproducibility_with_seed(self):
        """Ensure identical splits are generated when using the same seed."""
        for i in range(1, 11):
            self._create_dummy_image(f"receipt_{i:02d}.jpg")

        rows = [
            {
                "img_id": f"receipt_{i:02d}.jpg",
                "anno_num": "1",
                "anno_image_quality": "0.90",
                "anno_texts": f"Store {i}",
                "anno_labels": "SELLER",
                "anno_polygons": "[{'category_id': 15, 'bbox': [10, 10, 100, 20], 'width': 1000, 'height': 1000, 'segmentation': []}]",
            }
            for i in range(1, 11)
        ]
        self._write_csv(rows)

        out_run1 = self.base_path / "out_run1"
        out_run2 = self.base_path / "out_run2"

        prepare_dataset(self.csv_path, self.image_dir, out_run1, random_seed=42)
        prepare_dataset(self.csv_path, self.image_dir, out_run2, random_seed=42)

        content1 = (out_run1 / "mcocr_manifest_split.csv").read_text(encoding="utf-8")
        content2 = (out_run2 / "mcocr_manifest_split.csv").read_text(encoding="utf-8")
        self.assertEqual(content1, content2)

        summary1 = (out_run1 / "dataset_summary.json").read_text(encoding="utf-8")
        summary2 = (out_run2 / "dataset_summary.json").read_text(encoding="utf-8")
        self.assertEqual(summary1, summary2)

    def test_leakage_detection_raises_error(self):
        """Verify that check_data_leakage raises DataLeakageError when overlap occurs."""
        split_with_leakage = {
            "train": ["img_01.jpg", "img_02.jpg", "img_03.jpg"],
            "validation": ["img_03.jpg", "img_04.jpg"],
            "test": ["img_05.jpg"],
        }
        with self.assertRaises(DataLeakageError):
            check_data_leakage(split_with_leakage)

        split_valid = {
            "train": ["img_01.jpg", "img_02.jpg"],
            "validation": ["img_03.jpg"],
            "test": ["img_04.jpg"],
        }
        with self.assertRaises(DataLeakageError):
            check_data_leakage(
                split_valid,
                expected_img_ids=["img_01.jpg", "img_02.jpg", "img_03.jpg", "img_04.jpg", "img_05.jpg"],
            )

    def test_manifest_leakage_detection(self):
        """Verify check_manifest_leakage flags rows where the same img_id has different splits."""
        valid_rows = [
            {"img_id": "img1.jpg", "split": "train"},
            {"img_id": "img1.jpg", "split": "train"},
            {"img_id": "img2.jpg", "split": "validation"},
        ]
        self.assertTrue(check_manifest_leakage(valid_rows))

        leaky_rows = [
            {"img_id": "img1.jpg", "split": "train"},
            {"img_id": "img1.jpg", "split": "test"},
        ]
        with self.assertRaises(DataLeakageError):
            check_manifest_leakage(leaky_rows)

    def test_no_crops_created(self):
        """Ensure no image crops or intermediate image files are generated."""
        for i in range(1, 6):
            self._create_dummy_image(f"receipt_{i:02d}.jpg")

        rows = [
            {
                "img_id": f"receipt_{i:02d}.jpg",
                "anno_num": "1",
                "anno_image_quality": "0.90",
                "anno_texts": "Shop",
                "anno_labels": "SELLER",
                "anno_polygons": "[{'category_id': 15, 'bbox': [10, 10, 50, 20], 'width': 1000, 'height': 1000, 'segmentation': []}]",
            }
            for i in range(1, 6)
        ]
        self._write_csv(rows)

        images_before = set(self.image_dir.iterdir())
        prepare_dataset(self.csv_path, self.image_dir, self.output_dir, random_seed=42)
        images_after = set(self.image_dir.iterdir())

        self.assertEqual(images_before, images_after)

        output_files = {p.name for p in self.output_dir.iterdir()}
        self.assertEqual(
            output_files,
            {"mcocr_clean_manifest.csv", "mcocr_manifest_split.csv", "dataset_summary.json"},
        )

    def test_vietnamese_text_handling(self):
        """Ensure Vietnamese diacritics and symbols are preserved without encoding issues."""
        self._create_dummy_image("receipt_vi.jpg")
        rows = [
            {
                "img_id": "receipt_vi.jpg",
                "anno_num": "2",
                "anno_image_quality": "0.95",
                "anno_texts": "Cửa Hàng Quần Áo Trẻ Em|||Tổng cộng: 250.000 đ",
                "anno_labels": "SELLER|||TOTAL_COST",
                "anno_polygons": (
                    "[{'category_id': 15, 'bbox': [10, 10, 200, 30], 'width': 1000, 'height': 1000, 'segmentation': []},"
                    " {'category_id': 18, 'bbox': [10, 80, 150, 30], 'width': 1000, 'height': 1000, 'segmentation': []}]"
                ),
            }
        ]
        self._write_csv(rows)

        summary = prepare_dataset(self.csv_path, self.image_dir, self.output_dir, random_seed=42)
        self.assertEqual(summary["valid_invoices_used"], 1)
        self.assertEqual(summary["annotated_regions"], 2)

        with open(self.output_dir / "mcocr_clean_manifest.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        self.assertEqual(records[0]["text"], "Cửa Hàng Quần Áo Trẻ Em")
        self.assertEqual(records[1]["text"], "Tổng cộng: 250.000 đ")

    def test_alias_prepare_mcocr_dataset(self):
        """Verify the prepare_mcocr_dataset alias is callable and behaves identically."""
        self._create_dummy_image("receipt_01.jpg")
        rows = [
            {
                "img_id": "receipt_01.jpg",
                "anno_num": "1",
                "anno_image_quality": "0.90",
                "anno_texts": "Store",
                "anno_labels": "SELLER",
                "anno_polygons": "[{'category_id': 15, 'bbox': [10, 10, 50, 20], 'width': 1000, 'height': 1000, 'segmentation': []}]",
            }
        ]
        self._write_csv(rows)

        summary = prepare_mcocr_dataset(self.csv_path, self.image_dir, self.output_dir, random_seed=42)
        self.assertEqual(summary["valid_invoices_used"], 1)

    def test_empty_csv(self):
        """Verify preparation on empty dataset."""
        self._write_csv([])
        summary = prepare_dataset(self.csv_path, self.image_dir, self.output_dir)
        self.assertEqual(summary["total_original_records"], 0)
        self.assertEqual(summary["valid_invoices_used"], 0)
        self.assertEqual(summary["annotated_regions"], 0)
        self.assertEqual(summary["leakage_check_result"], "PASSED")


if __name__ == "__main__":
    unittest.main()
