import os
import tempfile
import unittest

import numpy as np

from src.evaluation.metrics import normalize_amount
from src.evaluation.offline_decode import decode_from_probs
from src.evaluation.probs_io import load_probs, save_probs
from src.preprocessing.token_dataset import LABEL2ID, LABELS

WORDS = ["MINIMART", "ANAN", "Ngay", "11/08/2020", "Tong", "tien:", "74,000", "Cam", "on"]
TAGS = ["B-SELLER", "I-SELLER", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST", "I-TOTAL_COST", "O", "O"]


def onehot(tags):
    probs = np.full((len(tags), len(LABELS)), 0.01 / (len(LABELS) - 1))
    for i, t in enumerate(tags):
        probs[i, LABEL2ID[t]] = 0.99
    return probs


class TestProbsIo(unittest.TestCase):
    def test_roundtrip_keeps_values_and_dtype(self):
        a, b = np.random.default_rng(0).random((3, 9)), np.zeros((0, 9))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.npz")
            save_probs(path, ["a.jpg", "b.jpg"], [a, b])
            loaded = load_probs(path)
        self.assertEqual(list(loaded), ["a.jpg", "b.jpg"])
        self.assertTrue((loaded["a.jpg"] == a).all())
        self.assertEqual(loaded["a.jpg"].dtype, np.float64)
        self.assertEqual(loaded["b.jpg"].shape, (0, 9))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            save_probs("unused.npz", ["a"], [])


class TestOfflineDecode(unittest.TestCase):
    def setUp(self):
        self.records = [{"img_id": "r1.jpg", "words": WORDS}]

    def test_decodes_like_the_extractor(self):
        preds, merged = decode_from_probs(self.records, {"r1.jpg": onehot(TAGS)})
        self.assertEqual(preds["r1.jpg"]["SELLER"], "MINIMART ANAN")
        self.assertEqual(normalize_amount(preds["r1.jpg"]["TOTAL_COST"]), 74000)
        self.assertIn("11/08/2020", preds["r1.jpg"]["TIMESTAMP"])
        self.assertIsNone(preds["r1.jpg"]["ADDRESS"])
        self.assertEqual(merged[0]["probs"].shape, (len(WORDS), len(LABELS)))
        self.assertTrue(merged[0]["covered"].all())

    def test_missing_probabilities(self):
        with self.assertRaises(KeyError):
            decode_from_probs(self.records, {})

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            decode_from_probs(self.records, {"r1.jpg": onehot(TAGS[:-1])})


if __name__ == "__main__":
    unittest.main()
