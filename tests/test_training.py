import unittest

import numpy as np

from src.training.seqmetrics import bio_spans, entity_scores
from src.training.windows import make_windows, merge_window_probs


class TestMakeWindows(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(make_windows([]), [])

    def test_short_receipt_is_one_window(self):
        self.assertEqual(make_windows([50, 50, 50], max_tokens=500), [(0, 3)])

    def test_windows_overlap_and_cover_everything(self):
        self.assertEqual(make_windows([100] * 10, max_tokens=350, overlap_tokens=100),
                         [(0, 3), (2, 5), (4, 7), (6, 9), (8, 10)])

    def test_word_larger_than_the_budget_gets_its_own_window(self):
        self.assertEqual(make_windows([10, 900, 10], max_tokens=500, overlap_tokens=100), [(0, 1), (1, 2), (2, 3)])

    def test_budget_respected_and_progress_guaranteed(self):
        counts = [1 + (i * 7) % 13 for i in range(300)]
        windows = make_windows(counts, max_tokens=60, overlap_tokens=15)
        covered = set()
        starts = [s for s, _ in windows]
        self.assertEqual(starts, sorted(set(starts)))                      # strictly increasing
        for s, e in windows:
            self.assertLess(s, e)
            self.assertLessEqual(sum(counts[s:e]), 60)
            covered.update(range(s, e))
        self.assertEqual(covered, set(range(300)))
        self.assertEqual(windows[-1][1], 300)


class TestMergeWindowProbs(unittest.TestCase):
    def test_overlap_is_averaged(self):
        w1 = (0, np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        w2 = (2, np.array([[1.0, 0.0], [0.0, 1.0]]))
        probs, covered = merge_window_probs(4, 2, [w1, w2])
        np.testing.assert_allclose(probs, [[1, 0], [1, 0], [0.5, 0.5], [0, 1]])
        self.assertTrue(covered.all())

    def test_uncovered_words_become_o(self):
        probs, covered = merge_window_probs(3, 2, [(0, np.array([[0.2, 0.8], [0.2, 0.8]]))])
        self.assertEqual(covered.tolist(), [True, True, False])
        np.testing.assert_allclose(probs[2], [1.0, 0.0])

    def test_truncated_window_and_no_windows(self):
        probs, covered = merge_window_probs(2, 2, [(1, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))])
        self.assertEqual(covered.tolist(), [False, True])
        probs, covered = merge_window_probs(2, 2, [])
        self.assertFalse(covered.any())


class TestBioSpans(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(bio_spans(["B-SELLER", "I-SELLER", "O", "B-TOTAL_COST"]),
                         {("SELLER", 0, 2), ("TOTAL_COST", 3, 4)})

    def test_inside_without_begin_starts_an_entity(self):
        self.assertEqual(bio_spans(["I-SELLER", "I-SELLER"]), {("SELLER", 0, 2)})

    def test_begin_splits_and_field_change_splits(self):
        self.assertEqual(bio_spans(["B-ADDRESS", "B-ADDRESS"]), {("ADDRESS", 0, 1), ("ADDRESS", 1, 2)})
        self.assertEqual(bio_spans(["B-SELLER", "I-ADDRESS"]), {("SELLER", 0, 1), ("ADDRESS", 1, 2)})

    def test_empty(self):
        self.assertEqual(bio_spans([]), set())


class TestEntityScores(unittest.TestCase):
    def test_missing_entity_lowers_recall(self):
        s = entity_scores([["B-SELLER", "I-SELLER", "O", "B-TOTAL_COST"]], [["B-SELLER", "I-SELLER", "O", "O"]])
        self.assertEqual(s["SELLER"]["f1"], 1.0)
        self.assertEqual(s["TOTAL_COST"]["recall"], 0.0)
        self.assertEqual(s["micro"]["precision"], 1.0)
        self.assertAlmostEqual(s["micro"]["recall"], 0.5)

    def test_boundary_error_counts_as_wrong(self):
        s = entity_scores([["B-SELLER", "I-SELLER", "O", "B-TOTAL_COST"]], [["B-SELLER", "O", "O", "B-TOTAL_COST"]])
        self.assertEqual((s["SELLER"]["tp"], s["SELLER"]["fp"], s["SELLER"]["fn"]), (0, 1, 1))
        self.assertEqual(s["TOTAL_COST"]["f1"], 1.0)
        self.assertAlmostEqual(s["micro"]["f1"], 0.5)

    def test_sums_over_receipts_and_all_o(self):
        s = entity_scores([["B-SELLER"], ["O"]], [["B-SELLER"], ["B-SELLER"]])
        self.assertEqual((s["SELLER"]["tp"], s["SELLER"]["fp"], s["SELLER"]["fn"]), (1, 1, 0))
        s = entity_scores([["O", "O"]], [["O", "O"]])
        self.assertEqual(s["micro"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
