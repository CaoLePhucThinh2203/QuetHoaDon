import unittest

from src.evaluation.metrics import field_match
from src.extraction.decode import assemble_fields, decode_entities, pick_amount, predict_fields


class TestDecodeEntities(unittest.TestCase):
    def test_groups_consecutive_words(self):
        words = ["Cua", "hang", "ABC", "Dia", "chi", "x"]
        tags = ["B-SELLER", "I-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "O"]
        ents = decode_entities(words, tags)
        self.assertEqual([(e["field"], e["start"], e["end"], e["text"]) for e in ents],
                         [("SELLER", 0, 3, "Cua hang ABC"), ("ADDRESS", 3, 5, "Dia chi")])

    def test_inside_tag_without_begin_starts_an_entity(self):
        ents = decode_entities(["a", "b"], ["I-SELLER", "I-SELLER"])
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0]["text"], "a b")

    def test_inside_tag_of_another_field_starts_a_new_entity(self):
        ents = decode_entities(["a", "b"], ["B-SELLER", "I-ADDRESS"])
        self.assertEqual([e["field"] for e in ents], ["SELLER", "ADDRESS"])

    def test_begin_tag_splits_same_field(self):
        ents = decode_entities(["a", "b"], ["B-ADDRESS", "B-ADDRESS"])
        self.assertEqual(len(ents), 2)

    def test_score_is_the_mean_over_the_entity(self):
        ents = decode_entities(["a", "b", "c"], ["B-SELLER", "I-SELLER", "O"], scores=[0.5, 1.0, 0.1])
        self.assertAlmostEqual(ents[0]["score"], 0.75)

    def test_empty_input(self):
        self.assertEqual(decode_entities([], []), [])


class TestPickAmount(unittest.TestCase):
    def test_plain_case(self):
        self.assertEqual(pick_amount(["Tong", "tien:", "74,000"]), "74,000")

    def test_currency_suffix_is_removed(self):
        self.assertEqual(pick_amount(["Thanh", "toan", "107,000d"]), "107,000")
        self.assertEqual(pick_amount(["VND74.000"]), "74.000")

    def test_split_thousands_are_rejoined(self):
        self.assertEqual(pick_amount(["Tong", "cong", "74", "000"]), "74,000")

    def test_separated_amount_beats_bare_digits(self):
        self.assertEqual(pick_amount(["0902", "74,000"]), "74,000")

    def test_bare_digits_are_used_when_nothing_else(self):
        self.assertEqual(pick_amount(["Tong", "74000"]), "74000")

    def test_last_amount_wins(self):
        self.assertEqual(pick_amount(["10,000", "74,000"]), "74,000")

    def test_no_amount(self):
        self.assertIsNone(pick_amount(["Tong", "tien"]))
        self.assertIsNone(pick_amount([]))


class TestAssembleFields(unittest.TestCase):
    def test_full_receipt(self):
        words = ["MINI", "MART", "Cho", "Sui", "Gia", "Lam", "Ngay", "11/08/2020", "Tong", "tien:", "74,000", "cam", "on"]
        tags = ["B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS",
                "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST", "O", "O"]
        self.assertEqual(predict_fields(words, tags), {
            "SELLER": "MINI MART",
            "ADDRESS": "Cho Sui Gia Lam",
            "TIMESTAMP": "Ngay 11/08/2020",
            "TOTAL_COST": "74,000",
        })

    def test_nothing_found(self):
        self.assertEqual(predict_fields(["a", "b"], ["O", "O"]),
                         {"SELLER": None, "ADDRESS": None, "TIMESTAMP": None, "TOTAL_COST": None})

    def test_address_pieces_are_joined_in_reading_order(self):
        words = ["a", "x", "b"]
        tags = ["B-ADDRESS", "O", "B-ADDRESS"]
        self.assertEqual(predict_fields(words, tags)["ADDRESS"], "a b")

    def test_seller_uses_the_most_confident_entity(self):
        words = ["a", "x", "b"]
        tags = ["B-SELLER", "O", "B-SELLER"]
        self.assertEqual(predict_fields(words, tags, scores=[0.4, 0.0, 0.9])["SELLER"], "b")

    def test_total_uses_the_most_confident_entity_with_an_amount(self):
        words = ["5,000", "x", "74,000", "y", "tien"]
        tags = ["B-TOTAL_COST", "O", "B-TOTAL_COST", "O", "B-TOTAL_COST"]
        scores = [0.6, 0.0, 0.9, 0.0, 0.99]
        self.assertEqual(predict_fields(words, tags, scores)["TOTAL_COST"], "74,000")

    def test_total_entity_without_a_number_gives_none(self):
        self.assertIsNone(predict_fields(["Tong", "tien"], ["B-TOTAL_COST", "I-TOTAL_COST"])["TOTAL_COST"])

    def test_unknown_fields_are_ignored(self):
        ents = [{"field": "ITEM_NAME", "start": 0, "end": 1, "words": ["a"], "text": "a", "score": 1.0}]
        self.assertEqual(assemble_fields(ents)["SELLER"], None)

    def test_prediction_matches_the_annotation_convention(self):
        # MC-OCR labels "Tong tien:" and the amount as separate TOTAL_COST regions.
        pred = predict_fields(["Tong", "tien:", "74,000"], ["B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST"])
        self.assertTrue(field_match("TOTAL_COST", pred["TOTAL_COST"], ["Tổng tiền:", "74,000"]))


if __name__ == "__main__":
    unittest.main()
