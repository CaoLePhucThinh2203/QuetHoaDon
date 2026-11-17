import unittest

import numpy as np

from src.extraction.field_cleanup import clean_field, clean_predictions, dedupe_tokens, strip_trailing_junk
from src.pipeline.extractor import fields_from_probs
from src.preprocessing.token_dataset import LABEL2ID, LABELS


class TestDedupe(unittest.TestCase):
    def test_examples_from_validation(self):
        self.assertEqual(dedupe_tokens("VM+ QNH 112 112 Thanh Niên"), "VM+ QNH 112 Thanh Niên")
        self.assertEqual(dedupe_tokens("ĐC: 212 Đường Đường Trần Phú"), "ĐC: 212 Đường Trần Phú")
        self.assertEqual(dedupe_tokens("Tân Lập 4 P. P.Cẩm Thủy"), "Tân Lập 4 P.Cẩm Thủy")
        self.assertEqual(dedupe_tokens("P.11, Q. Q.PN"), "P.11, Q.PN")
        self.assertEqual(dedupe_tokens("Saigon Co.op op Co.op .op Food THE K- K-PARK"), "Saigon Co.op Co.op Food THE K-PARK")

    def test_keeps_normal_text(self):
        for t in ("Tổ 7, khu Minh Tiến A Tổ 7, khu Minh Tiến A", "1060-1062 Trần Phú", "P. Cẩm Tây", "Q.3 TP.HCM"):
            self.assertEqual(dedupe_tokens(t), t)
        self.assertIsNone(dedupe_tokens(None))


class TestTrailing(unittest.TestCase):
    def test_address(self):
        self.assertEqual(strip_trailing_junk("Cẩm Phả, Quảng Ninh 024", "ADDRESS"), "Cẩm Phả, Quảng Ninh")
        self.assertEqual(strip_trailing_junk("Gia Lâm Hà Nội 0945984733", "ADDRESS"), "Gia Lâm Hà Nội")
        self.assertEqual(strip_trailing_junk("Cẩm Phả - QN Tel:", "ADDRESS"), "Cẩm Phả - QN")
        self.assertEqual(strip_trailing_junk("Tinh Bình Duong Ma so", "ADDRESS"), "Tinh Bình Duong")
        self.assertEqual(strip_trailing_junk("Nam Từ Liêm ĐT(Tel)", "ADDRESS"), "Nam Từ Liêm")
        self.assertEqual(strip_trailing_junk("168 Nguyen Dinh Chieu Q.3", "ADDRESS"), "168 Nguyen Dinh Chieu Q.3")

    def test_seller(self):
        self.assertEqual(strip_trailing_junk("SIEU THI BACH HOA TONG HOP Số", "SELLER"), "SIEU THI BACH HOA TONG HOP")
        self.assertEqual(strip_trailing_junk("THE COFFEE HQUSE 403", "SELLER"), "THE COFFEE HQUSE")
        self.assertEqual(strip_trailing_junk("VINMART 4967.", "SELLER"), "VINMART")
        self.assertEqual(strip_trailing_junk("Tel", "SELLER"), "Tel")          # never empties the field

    def test_short_numbers_stay_in_addresses(self):
        self.assertEqual(strip_trailing_junk("Quận 9", "ADDRESS"), "Quận 9")


class TestCleanPredictions(unittest.TestCase):
    def test_only_seller_and_address(self):
        preds = {"a": {"SELLER": "X Số", "ADDRESS": "Y 112 112 Z", "TOTAL_COST": "74,000 5", "TIMESTAMP": None}}
        out = clean_predictions(preds)
        self.assertEqual(out["a"], {"SELLER": "X", "ADDRESS": "Y 112 Z", "TOTAL_COST": "74,000 5", "TIMESTAMP": None})
        self.assertEqual(preds["a"]["SELLER"], "X Số")                     # input not modified
        self.assertEqual(clean_field("TOTAL_COST", "1 1"), "1 1")


def onehot(tags):
    p = np.full((len(tags), len(LABELS)), 0.01 / (len(LABELS) - 1))
    for i, t in enumerate(tags):
        p[i, LABEL2ID[t]] = 0.99
    return p


class TestExtractorPostprocess(unittest.TestCase):
    def test_off_by_default_and_on_when_asked(self):
        words = ["SIEU", "THI", "Số", "Ngay", "11/08/2020", "20:50"]
        tags = ["B-SELLER", "I-SELLER", "I-SELLER", "O", "B-TIMESTAMP", "O"]
        rec = {"img_id": "r", "words": words}
        plain, _ = fields_from_probs(rec, onehot(tags))
        post, _ = fields_from_probs(rec, onehot(tags), postprocess=True)
        self.assertEqual(plain["SELLER"]["text"], "SIEU THI Số")
        self.assertEqual(post["SELLER"]["text"], "SIEU THI")
        self.assertEqual(post["SELLER"]["source"], "model")
        self.assertEqual(post["TIMESTAMP"]["text"], "11/08/2020 20:50")


if __name__ == "__main__":
    unittest.main()
