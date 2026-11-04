"""Tests for the canonical extraction output schema."""
import unittest

from src.extraction.schema import (
    TargetEntity,
    REQUIRED_OUTPUT_FIELDS,
    OPTIONAL_OUTPUT_FIELDS,
    ANNOTATED_FIELDS,
    empty_field,
    empty_receipt_result,
)


class TestSchema(unittest.TestCase):
    def test_required_fields(self):
        self.assertEqual(
            REQUIRED_OUTPUT_FIELDS,
            ["SELLER", "ADDRESS", "ITEM_NAME", "ITEM_PRICE", "TOTAL_COST"],
        )

    def test_optional_fields(self):
        self.assertIn("QUANTITY", OPTIONAL_OUTPUT_FIELDS)
        self.assertIn("TIMESTAMP", OPTIONAL_OUTPUT_FIELDS)

    def test_annotated_fields_match_mcocr(self):
        self.assertEqual(set(ANNOTATED_FIELDS), {"SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"})

    def test_empty_field_is_explicit(self):
        self.assertEqual(
            empty_field(),
            {"text": None, "bbox": None, "confidence": None, "status": "not_found"},
        )

    def test_empty_receipt_structure(self):
        result = empty_receipt_result()
        self.assertIn("seller", result)
        self.assertIn("address", result)
        self.assertEqual(result["items"], [])
        self.assertIn("total", result)
        self.assertIn("timestamp", result)


if __name__ == "__main__":
    unittest.main()
