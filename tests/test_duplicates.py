import unittest

from src.preprocessing.duplicates import find_duplicates, gold_key, jaccard, leaking_receipts, word_set


def rec(fid, words, seller="MINI MART", date="14/08/2020", total="74,000"):
    return {"file_id": fid, "img_id": fid + ".jpg", "image": fid + ".jpg", "words": words,
            "gt": {"SELLER": [seller], "TIMESTAMP": ["Ngày " + date], "TOTAL_COST": ["Tổng", total]}}


BASE = [f"hang{i:02d}" for i in range(30)]


class TestDuplicates(unittest.TestCase):
    def test_gold_key_needs_all_three_fields(self):
        self.assertEqual(gold_key(rec("a", BASE))[:2], ("14/08/2020", 74000))
        r = rec("b", BASE); r["gt"].pop("SELLER")
        self.assertIsNone(gold_key(r))

    def test_jaccard(self):
        self.assertEqual(jaccard(frozenset("abc"), frozenset("abc")), 1.0)
        self.assertEqual(jaccard(frozenset("ab"), frozenset("cd")), 0.0)
        self.assertEqual(jaccard(frozenset(), frozenset("a")), 0.0)

    def test_pairs_across_splits_are_flagged_by_the_right_signal(self):
        splits = {"train": [rec("t1", BASE), rec("t2", [f"x{i:02d}" for i in range(30)], seller="OTHER", total="5,000")],
                  "test": [rec("e1", BASE + ["extra"]), rec("e2", [f"y{i:02d}" for i in range(30)], seller="THIRD", total="9,000")]}
        pairs = find_duplicates(splits)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(sorted((pairs[0]["a"], pairs[0]["b"])), ["e1", "t1"])
        self.assertEqual(set(pairs[0]["signals"]), {"gold", "text"})
        leak = leaking_receipts(pairs)
        self.assertEqual(leak["e1"], ["t1"])

    def test_same_split_pairs_do_not_leak(self):
        pairs = find_duplicates({"train": [rec("a", BASE), rec("b", BASE)]})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(leaking_receipts(pairs), {})


if __name__ == "__main__":
    unittest.main()
