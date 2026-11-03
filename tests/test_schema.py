"""Unit tests for entity classification schema and BIO tag generation."""
import unittest

from src.extraction.schema import (
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
    TargetEntity,
    get_bio_label,
)


class TestSchema(unittest.TestCase):

    def test_schema_cardinality(self):
        # 1 'O' + 4 entities * 2 (B-, I-) = 9 labels
        self.assertEqual(len(BIO_LABELS), 9)
        self.assertEqual(len(LABEL2ID), 9)
        self.assertEqual(len(ID2LABEL), 9)

    def test_bijective_mapping(self):
        for label, idx in LABEL2ID.items():
            self.assertEqual(ID2LABEL[idx], label)

    def test_get_bio_label(self):
        self.assertEqual(get_bio_label("SELLER", is_first_token=True), "B-SELLER")
        self.assertEqual(get_bio_label("SELLER", is_first_token=False), "I-SELLER")
        self.assertEqual(get_bio_label("TOTAL_COST", is_first_token=True), "B-TOTAL_COST")
        self.assertEqual(get_bio_label("TOTAL_COST", is_first_token=False), "I-TOTAL_COST")
        self.assertEqual(get_bio_label("O", is_first_token=True), "O")
        self.assertEqual(get_bio_label("OTHER", is_first_token=True), "O")
        self.assertEqual(get_bio_label(None), "O")


if __name__ == "__main__":
    unittest.main()
