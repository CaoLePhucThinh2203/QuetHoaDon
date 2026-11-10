"""Split-amount rejoining, rules, bootstrap, error causes and the B0 baseline (pure NumPy / scikit-learn)."""
import unittest

import numpy as np

from src.extraction.decode import amount_candidates, merge_split_amount, pick_amount
from src.extraction.rules import fill_missing, fold, rule_timestamp, rule_total
from src.preprocessing.token_dataset import LABEL2ID
from src.training import field_eval as fe


def probs_row(**mass):
    row = np.zeros(len(LABEL2ID))
    for lab, p in mass.items():
        row[LABEL2ID[lab.replace("__", "-")]] = p
    row[0] = 1.0 - row.sum()
    return row


class TestSplitAmounts(unittest.TestCase):
    def test_pairs_seen_on_validation(self):
        cases = {("107,0", ",000"): 107000, ("10,0", "000"): 10000, ("253", "000"): 253000,
                 ("87,0", "L000"): 87000, ("442,1", "000"): 442000, ("10", "L000"): 10000,
                 ("260,", ",000"): 260000, ("97", "000d"): 97000, ("52", "700"): 52700, ("30,0", "1000"): 30000}
        for (h, t), v in cases.items():
            self.assertEqual(merge_split_amount(h, t), v, (h, t))

    def test_pairs_that_must_not_merge(self):
        for h, t in (("253", "1000"), ("12.000", "000"), ("Tong", "000"), ("0902", "123"), ("10,000", "74,000")):
            self.assertIsNone(merge_split_amount(h, t), (h, t))

    def test_candidates_span_both_words(self):
        c = amount_candidates(["Tong", "107,0", ",000", "x", "46,000"])
        self.assertEqual([(s, e, v) for s, e, _, v in c], [(1, 3, 107000), (4, 5, 46000)])

    def test_pick_amount_rejoins(self):
        self.assertEqual(pick_amount(["Tong", "cong", "168,0", ",000"]), "168,000")
        self.assertEqual(pick_amount(["Tong", "cong", "74", "000"]), "74,000")

    def test_soft_total_uses_the_rejoined_amount(self):
        words = ["Tong", "107,0", ",000"]
        probs = np.stack([probs_row(), probs_row(I__TOTAL_COST=0.01), probs_row(I__TOTAL_COST=0.6)])
        self.assertEqual(fe.soft_total(words, probs), "107,000")


class TestRules(unittest.TestCase):
    def test_fold(self):
        self.assertEqual(fold("Tổng cộng:"), "tongcong")

    def test_total_after_keyword_skips_cash_given(self):
        words = ["Sua", "12,000", "Tổng", "cộng:", "107,0", ",000", "Tiền", "khách", "đưa", "200,000"]
        self.assertEqual(rule_total(words), "107,000")
        self.assertIsNone(rule_total(["Sua", "12,000"]))

    def test_timestamp(self):
        self.assertEqual(rule_timestamp(["Ngay", "12-08-2020", "x"]), "12-08-2020")
        self.assertIsNone(rule_timestamp(["abc"]))

    def test_fill_missing_keeps_model_values(self):
        recs = [{"img_id": "a", "words": ["Tong", "cong", "50,000", "01/02/2020"]}]
        out = fill_missing(recs, {"a": {"SELLER": "X", "TOTAL_COST": "40,000", "TIMESTAMP": None}})
        self.assertEqual(out["a"]["TOTAL_COST"], "40,000")
        self.assertEqual(out["a"]["TIMESTAMP"], "01/02/2020")

    def test_date_rule_beats_the_model_but_not_a_missing_date(self):
        recs = [{"img_id": "a", "words": ["Ngay", "01/02/2020"]}, {"img_id": "b", "words": ["abc"]}]
        out = fill_missing(recs, {"a": {"TIMESTAMP": "xx"}, "b": {"TIMESTAMP": "05/05/2020"}})
        self.assertEqual(out["a"]["TIMESTAMP"], "01/02/2020")
        self.assertEqual(out["b"]["TIMESTAMP"], "05/05/2020")
        out = fill_missing(recs, {"a": {"TIMESTAMP": "xx"}}, prefer_rule=())
        self.assertEqual(out["a"]["TIMESTAMP"], "xx")

    def test_timestamp_without_a_date_is_dropped_and_split_dates_are_joined(self):
        recs = [{"img_id": "a", "words": ["Ngay", "14/08/2", "2020", "08:29"]}, {"img_id": "b", "words": ["abc"]}]
        out = fill_missing(recs, {"a": {}, "b": {"TIMESTAMP": "Ngay ban: junk"}})
        self.assertEqual(out["a"]["TIMESTAMP"], "14/08/2 2020")
        self.assertIsNone(out["b"]["TIMESTAMP"])
        self.assertEqual(fe.normalize_date("Ngày bân: 13/08/2 2020 10: 19"), "13/08/2020")
        self.assertEqual(fe.normalize_date("14/08/2020 08:29"), "14/08/2020")

    def test_gt_amount_ignores_barcodes(self):
        self.assertEqual(fe.gt_amount(["Tong", "74,000", "0902543800123"]), 74000)
        self.assertIsNone(fe.gt_amount(["0902543800123"]))


def rec(name, words, labels, gt):
    return {"img_id": name + ".jpg", "file_id": name, "words": words, "labels": labels, "gt": gt,
            "boxes": [[10 * i, 10 * i, 10 * i + 8, 10 * i + 8] for i in range(len(words))]}


class TestBootstrapAndErrors(unittest.TestCase):
    def test_bootstrap_interval_contains_the_point(self):
        gts = {f"r{i}": {"TOTAL_COST": ["50,000"]} for i in range(20)}
        preds = {f"r{i}": {"TOTAL_COST": "50,000" if i % 2 else "1,000"} for i in range(20)}
        ci = fe.bootstrap_ci(fe.receipt_outcomes(preds, gts), n_boot=200)
        self.assertAlmostEqual(ci["TOTAL_COST"]["value"], 0.5)
        self.assertLessEqual(ci["TOTAL_COST"]["low"], 0.5)
        self.assertGreaterEqual(ci["TOTAL_COST"]["high"], 0.5)

    def test_paired_bootstrap_sees_a_better_system(self):
        gts = {f"r{i}": {"TOTAL_COST": ["50,000"]} for i in range(30)}
        a = fe.receipt_outcomes({k: {"TOTAL_COST": "1,000"} for k in gts}, gts)
        b = fe.receipt_outcomes({k: {"TOTAL_COST": "50,000"} for k in gts}, gts)
        rows = fe.per_receipt([{"img_id": k} for k in gts], {}, b)
        res = fe.paired_bootstrap(a, fe.outcomes_from_rows(rows), n_boot=100)
        self.assertEqual(res["receipts"], 30)
        self.assertGreater(res["metrics"]["TOTAL_COST"]["diff"], 0.9)
        self.assertEqual(res["metrics"]["TOTAL_COST"]["p_b_better"], 1.0)

    def test_error_causes(self):
        recs = [
            rec("ocr", ["Tong", "5,000"], ["B-TOTAL_COST", "I-TOTAL_COST"], {"TOTAL_COST": ["74,000"]}),
            rec("miss", ["Tong", "74,000"], ["B-TOTAL_COST", "I-TOTAL_COST"], {"TOTAL_COST": ["74,000"]}),
            rec("nolab", ["Tong", "74,000"], ["O", "O"], {"TOTAL_COST": ["74,000"]}),
            rec("noamt", ["Tong"], ["B-TOTAL_COST"], {"TOTAL_COST": ["Tong tien"]}),
        ]
        merged = [{"probs": np.stack([probs_row()] * len(r["words"]))} for r in recs]
        ec = fe.error_causes(recs, merged, {r["img_id"]: {} for r in recs})
        self.assertEqual(ec["by_field"]["TOTAL_COST"]["ocr_text_error"], 1)
        self.assertEqual(ec["by_field"]["TOTAL_COST"]["model_missed"], 1)
        self.assertEqual(ec["by_field"]["TOTAL_COST"]["no_ocr_word_labelled"], 1)
        self.assertEqual(ec["by_field"]["TOTAL_COST"]["annotation_without_amount"], 1)
        self.assertEqual(ec["receipts_with_error"], 4)

    def test_report_has_week4_sections(self):
        r = rec("a", ["MINI", "Tong", "74,000"], ["B-SELLER", "B-TOTAL_COST", "I-TOTAL_COST"],
                {"SELLER": ["MINI"], "TOTAL_COST": ["74,000"]})
        probs = np.zeros((3, len(LABEL2ID)))
        for i, t in enumerate(r["labels"]):
            probs[i, LABEL2ID[t]] = 1.0
        rep = fe.build_report([r], [{"probs": probs}], n_boot=20)
        self.assertEqual(rep["fields_oracle"]["TOTAL_COST"]["f1"], 1.0)
        self.assertEqual(rep["receipt_exact_match_lenient_soft_rules"]["rate"], 1.0)
        self.assertEqual(rep["bootstrap_soft_rules"]["macro_f1"]["value"], 0.5)   # 2 of 4 fields annotated
        self.assertIn("TOTAL_COST", fe.format_report(rep))


class TestBaselineB0(unittest.TestCase):
    def test_fit_and_predict_on_synthetic_words(self):
        from src.baseline.b0 import N_HANDCRAFTED, cluster_summary, fit_b0, handcrafted_features, predict_b0, silhouette

        rng = np.random.default_rng(0)
        recs, embs = [], []
        for d in range(12):
            words = ["SHOP", "Tong", "74,000"] + ["item"] * 5
            labels = ["B-SELLER", "B-TOTAL_COST", "I-TOTAL_COST"] + ["O"] * 5
            recs.append(rec(f"r{d}", words, labels, {"SELLER": ["SHOP"], "TOTAL_COST": ["74,000"]}))
            centers = {"B-SELLER": 0.0, "B-TOTAL_COST": 5.0, "I-TOTAL_COST": 10.0, "O": -5.0}
            embs.append(np.stack([np.full(16, centers[t]) + rng.normal(0, 0.1, 16) for t in labels]).astype(np.float32))
        self.assertEqual(handcrafted_features(recs[0]).shape, (8, N_HANDCRAFTED))
        model = fit_b0(embs[:8], recs[:8], k=4, pca_dim=4, hand_weight=0.0)   # embeddings alone separate the labels
        merged, x, c = predict_b0(model, embs[8:], recs[8:])
        self.assertEqual(merged[0]["probs"].shape, (8, len(LABEL2ID)))
        self.assertAlmostEqual(float(merged[0]["probs"].sum(axis=1)[0]), 1.0, places=6)
        self.assertGreater(silhouette(x, c), 0.3)
        self.assertEqual(len(cluster_summary(model)), 4)
        rep = fe.build_report(recs[8:], merged, n_boot=20)
        self.assertEqual(rep["fields_lenient_soft_rules"]["TOTAL_COST"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
