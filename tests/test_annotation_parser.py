"""Unit tests for MC-OCR dataset annotation parsing and validation."""
import unittest

from src.preprocessing.annotation_parser import (
    MCOCRRecord,
    parse_mcocr_row,
)


class TestAnnotationParser(unittest.TestCase):

    def test_parse_valid_row(self):
        row = {
            "img_id": "test_001.jpg",
            "anno_num": "2",
            "anno_image_quality": "0.85",
            "anno_texts": "Cua Hang ABC|||50,000 VND",
            "anno_labels": "SELLER|||TOTAL_COST",
            "anno_polygons": (
                "[{'category_id': 15, 'bbox': [100, 50, 200, 30], 'width': 1000, 'height': 1000, 'segmentation': []},"
                " {'category_id': 18, 'bbox': [100, 500, 150, 40], 'width': 1000, 'height': 1000, 'segmentation': []}]"
            ),
        }
        record = parse_mcocr_row(row)
        self.assertTrue(record.is_valid)
        self.assertEqual(record.anno_num, 2)
        self.assertEqual(record.img_id, "test_001.jpg")
        self.assertEqual(len(record.instances), 2)
        self.assertEqual(record.instances[0].label, "SELLER")
        self.assertEqual(record.instances[0].text, "Cua Hang ABC")
        self.assertEqual(record.instances[1].label, "TOTAL_COST")
        self.assertEqual(record.instances[1].text, "50,000 VND")

    def test_parse_anomaly_label_normalization(self):
        # Testing real anomaly found in raw dataset row 8: TOTAL_TOTAL_COST
        row = {
            "img_id": "test_anomaly.jpg",
            "anno_num": "1",
            "anno_image_quality": "0.90",
            "anno_texts": "Tổng cộng: 120,000",
            "anno_labels": "TOTAL_TOTAL_COST",
            "anno_polygons": (
                "[{'category_id': 18, 'bbox': [50, 800, 300, 40], 'width': 1000, 'height': 1000, 'segmentation': []}]"
            ),
        }
        record = parse_mcocr_row(row)
        self.assertTrue(record.is_valid)
        self.assertEqual(record.instances[0].label, "TOTAL_COST")

    def test_parse_empty_row(self):
        # Testing real empty row pattern from raw dataset rows 1011 & 1091
        row = {
            "img_id": "test_empty.jpg",
            "anno_num": "0",
            "anno_image_quality": "0.45",
            "anno_texts": "",
            "anno_labels": "",
            "anno_polygons": "[]",
        }
        record = parse_mcocr_row(row)
        self.assertTrue(record.is_valid)
        self.assertEqual(record.anno_num, 0)
        self.assertEqual(len(record.instances), 0)

    def test_parse_length_mismatch(self):
        row = {
            "img_id": "test_mismatch.jpg",
            "anno_num": "2",
            "anno_image_quality": "0.75",
            "anno_texts": "Only One Text",
            "anno_labels": "SELLER|||TOTAL_COST",
            "anno_polygons": "[]",
        }
        record = parse_mcocr_row(row)
        self.assertFalse(record.is_valid)
        self.assertIn("Length mismatch", record.error_message)


if __name__ == "__main__":
    unittest.main()
