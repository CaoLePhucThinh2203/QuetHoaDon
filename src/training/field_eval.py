"""Field-level evaluation of token-classifier predictions (pure Python / NumPy / scikit-learn).

Entity-level F1 (``seqmetrics``) is a harsh training signal: one wrongly cut word makes a whole entity wrong.
What downstream use needs is the *value of each field*, so this module decodes the predicted tags into
``SELLER / ADDRESS / TIMESTAMP / TOTAL_COST`` and scores them with the same rules as the rest of the project
(``src.evaluation.metrics.field_scores``), plus:

* receipt-level exact match (every annotated field of the receipt correct);
* error of the total amount in VND (MAE, RMSE, median absolute error);
* one-vs-rest ROC-AUC per label from the word probabilities;
* per-label word precision / recall / F1 and the word confusion matrix.
"""
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.metrics import (FIELDS, cer, field_match, field_scores, gold_amounts, normalize_amount,
                                    normalize_text, prf1)
from src.extraction.decode import amount_candidates, predict_fields
from src.extraction.rules import DATE_SPLIT_RE, fill_missing
from src.preprocessing.token_dataset import ID2LABEL, LABEL2ID, LABELS

_TOTAL_IDS = [LABEL2ID["B-TOTAL_COST"], LABEL2ID["I-TOTAL_COST"]]
_TIMESTAMP_IDS = [LABEL2ID["B-TIMESTAMP"], LABEL2ID["I-TIMESTAMP"]]
_DATE_RE = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})")
PLAUSIBLE_MIN, PLAUSIBLE_MAX = 1000, 500_000_000     # VND range of a receipt total
LENIENT_CER = 0.3           # SELLER / ADDRESS count as correct when the character error rate is at most this


def decode_predictions(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]]
                       ) -> Dict[str, Dict[str, Optional[str]]]:
    """``{img_id: {field: text or None}}`` from the merged word probabilities of every receipt."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for rec, m in zip(records, merged):
        probs = m["probs"]
        tags = [ID2LABEL[int(i)] for i in probs.argmax(axis=1)]
        scores = probs.max(axis=1).tolist()
        out[rec["img_id"]] = predict_fields(rec["words"], tags, scores)
    return out


def gold_fields(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    return {rec["img_id"]: rec.get("gt") or {} for rec in records}


def gt_amount(texts: Optional[Sequence[str]]) -> Optional[int]:
    """The annotated total: the largest amount among the TOTAL_COST region texts (the label text has none)."""
    amounts = [a for a in gold_amounts(texts) if a and PLAUSIBLE_MIN <= a <= PLAUSIBLE_MAX]   # barcodes are not totals
    return max(amounts) if amounts else None


def amount_errors(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    """MAE / RMSE / median absolute error of the predicted total (VND) on receipts where both exist."""
    diffs: List[float] = []
    n_gt = n_missing = 0
    for img_id, g in gts.items():
        true = gt_amount(g.get("TOTAL_COST"))
        if true is None:
            continue
        n_gt += 1
        pred = normalize_amount((preds.get(img_id) or {}).get("TOTAL_COST"))
        if pred is None:
            n_missing += 1
            continue
        diffs.append(float(pred - true))
    d = np.array(diffs, dtype=np.float64)
    has = len(d) > 0
    return {
        "receipts_with_gt_total": n_gt,
        "no_amount_predicted": n_missing,
        "compared": len(d),
        "exact": int((d == 0).sum()) if has else 0,
        "mae": float(np.abs(d).mean()) if has else None,
        "rmse": float(np.sqrt((d ** 2).mean())) if has else None,
        "median_abs_error": float(np.median(np.abs(d))) if has else None,
    }


def receipt_exact_match(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    """Share of receipts whose every annotated field is predicted correctly."""
    n = ok = 0
    for img_id, g in gts.items():
        present = [f for f in FIELDS if g.get(f)]
        if not present:
            continue
        n += 1
        p = preds.get(img_id) or {}
        if all(field_match(f, p.get(f), g[f]) for f in present):
            ok += 1
    return {"receipts": n, "all_fields_correct": ok, "rate": ok / n if n else 0.0}


def token_auc(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """One-vs-rest ROC-AUC per label over all words (``None`` when a label is absent from the gold tags)."""
    from sklearn.metrics import roc_auc_score

    y = np.array([LABEL2ID[t] for rec in records for t in rec["labels"]])
    p = np.concatenate([m["probs"] for m in merged], axis=0)
    out: Dict[str, Optional[float]] = {}
    for i, lab in enumerate(LABELS):
        yi = y == i
        out[lab] = float(roc_auc_score(yi, p[:, i])) if 0 < yi.sum() < len(yi) else None
    vals = [v for v in out.values() if v is not None]
    out["macro"] = float(np.mean(vals)) if vals else None
    return out


def word_confusion(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]]) -> np.ndarray:
    """``cm[gold, predicted]`` word counts over the label set (rows and columns follow ``LABELS``)."""
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for rec, m in zip(records, merged):
        pred = m["probs"].argmax(axis=1)
        for tag, pi in zip(rec["labels"], pred):
            cm[LABEL2ID[tag], int(pi)] += 1
    return cm


def token_prf(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for i, lab in enumerate(LABELS):
        tp = int(cm[i, i])
        out[lab] = prf1(tp, int(cm[:, i].sum()) - tp, int(cm[i, :].sum()) - tp)
    return out


def total_cost_errors(records: Sequence[Dict[str, Any]], preds: Dict[str, Dict[str, Optional[str]]],
                      limit: int = 50) -> List[Dict[str, Any]]:
    """Receipts whose total is wrong, for the error analysis (gold texts and the predicted amount)."""
    errors: List[Dict[str, Any]] = []
    for rec in records:
        gt = (rec.get("gt") or {}).get("TOTAL_COST")
        if not gt:
            continue
        pred = (preds.get(rec["img_id"]) or {}).get("TOTAL_COST")
        if not field_match("TOTAL_COST", pred, gt):
            errors.append({"file_id": rec["file_id"], "predicted": pred, "gold_texts": gt})
            if len(errors) >= limit:
                break
    return errors


# ---------------------------------------------------------------------------------- soft decoding
def normalize_date(text: Optional[str]) -> Optional[str]:
    """First ``dd/mm/yyyy`` date found in ``text`` (2-digit years become 20yy), or ``None``."""
    m = _DATE_RE.search(text or "") or DATE_SPLIT_RE.search(text or "")
    if not m:
        return None
    day, month, year = m.groups()
    if len(year) == 2:
        year = "20" + year
    return f"{int(day):02d}/{int(month):02d}/{year}"


def soft_total(words: Sequence[str], probs: np.ndarray, min_prob: float = 0.02) -> Optional[str]:
    """Amount-shaped word with the largest TOTAL_COST probability (``B-`` plus ``I-``), even if its argmax tag is O.

    Hard decoding found no amount for almost half of the validation receipts; a word that merely *looks* like
    the total (right format, plausible value) and has some TOTAL_COST mass is a better answer than nothing.
    Ties go to the later word (the total is near the bottom).
    """
    best = None
    for start, end, text, _ in amount_candidates(words):
        score = float(probs[start:end][:, _TOTAL_IDS].sum(axis=1).max())
        if best is None or score >= best[0]:
            best = (score, text)
    return best[1] if best is not None and best[0] >= min_prob else None


def soft_timestamp(words: Sequence[str], probs: np.ndarray, min_prob: float = 0.02) -> Optional[str]:
    """Date-shaped word with the largest TIMESTAMP probability, or ``None``."""
    best = None
    for i, w in enumerate(words):
        if normalize_date(w) is None:
            continue
        score = float(probs[i, _TIMESTAMP_IDS].sum())
        if best is None or score > best[0]:
            best = (score, w)
    return best[1] if best is not None and best[0] >= min_prob else None


def decode_predictions_soft(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]]
                            ) -> Dict[str, Dict[str, Optional[str]]]:
    """Hard decoding, with TOTAL_COST and TIMESTAMP replaced by the soft choice whenever one exists."""
    preds = decode_predictions(records, merged)
    for rec, m in zip(records, merged):
        p = preds[rec["img_id"]]
        total = soft_total(rec["words"], m["probs"])
        if total:
            p["TOTAL_COST"] = total
        stamp = soft_timestamp(rec["words"], m["probs"])
        if stamp:
            p["TIMESTAMP"] = stamp
    return preds


# ---------------------------------------------------------------------------------- total-amount diagnosis
def _amount_words(words: Sequence[str]) -> List[tuple]:
    """``(start, end, value)`` of every amount candidate, rejoining amounts that OCR split in two words."""
    return [(s, e, v) for s, e, _, v in amount_candidates(words)]


def total_cause(rec: Dict[str, Any], probs: np.ndarray, min_prob: float = 0.02) -> str:
    """Cause category of one receipt's total (see ``total_diagnostics``)."""
    truth = gt_amount((rec.get("gt") or {}).get("TOTAL_COST"))
    if truth is None:
        return "no_gt_amount"
    scored = [(float(probs[s:e][:, _TOTAL_IDS].sum(axis=1).max()), s, v) for s, e, v in _amount_words(rec["words"])]
    match = [x for x in scored if x[2] == truth]
    if not match:
        return "amount_not_in_ocr"
    best_all = max(scored, key=lambda x: (x[0], x[1]))
    best_match = max(match, key=lambda x: (x[0], x[1]))
    if best_match[0] < min_prob:
        return "below_min_prob"
    return "top1" if best_match[:2] == best_all[:2] else "ranked_lower"


def total_diagnostics(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]],
                      min_prob: float = 0.02, example_limit: int = 10) -> Dict[str, Any]:
    """Why is the total missing? Splits the receipts with an annotated total into causes.

    * ``amount_not_in_ocr``: no OCR word (or rejoined pair) equals the annotated amount (OCR / parsing ceiling);
    * ``below_min_prob``: the right word exists but the model gives it almost no TOTAL_COST probability;
    * ``ranked_lower``: the right word exists with some probability but another amount-shaped word has more;
    * ``top1``: the right word has the highest TOTAL_COST probability among amount-shaped words.
    """
    cats = {"no_gt_amount": 0, "amount_not_in_ocr": 0, "below_min_prob": 0, "ranked_lower": 0, "top1": 0}
    examples: Dict[str, List[str]] = {k: [] for k in cats}
    for rec, m in zip(records, merged):
        cat = total_cause(rec, m["probs"], min_prob)
        cats[cat] += 1
        if cat != "no_gt_amount" and len(examples[cat]) < example_limit:
            examples[cat].append(rec.get("file_id", rec.get("img_id")))
    return {"categories": cats, "examples": examples}


# ---------------------------------------------------------------------------------- lenient scoring
def lenient_match(field: str, pred: Optional[str], gt_texts: Optional[Sequence[str]]) -> bool:
    """Match rules that tolerate OCR noise in the annotated *text*.

    The annotations are human transcriptions while predictions are OCR output, so exact string equality is capped
    by the OCR error rate. SELLER / ADDRESS: character error rate <= ``LENIENT_CER``; TIMESTAMP: same calendar
    date; TOTAL_COST: same amount (identical to the strict rule).
    """
    if not pred or not gt_texts:
        return False
    if field == "TOTAL_COST":
        return field_match(field, pred, list(gt_texts))
    if field == "TIMESTAMP":
        d = normalize_date(pred)
        return d is not None and any(normalize_date(g) == d for g in gt_texts)
    return cer(normalize_text(" ".join(gt_texts)), normalize_text(pred)) <= LENIENT_CER


def lenient_scores(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]],
                   fields: Sequence[str] = FIELDS) -> Dict[str, Dict[str, float]]:
    """Per-field P/R/F1 with ``lenient_match``; counting rules are the same as ``field_scores``."""
    out: Dict[str, Dict[str, float]] = {}
    for fld in fields:
        tp = fp = fn = n_gt = 0
        for img_id, g in gts.items():
            gt = g.get(fld)
            pred = (preds.get(img_id) or {}).get(fld)
            n_gt += 1 if gt else 0
            if pred and gt:
                if lenient_match(fld, pred, gt):
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif pred and not gt:
                fp += 1
            elif gt and not pred:
                fn += 1
        res = prf1(tp, fp, fn)
        res["exact_match"] = tp / n_gt if n_gt else 0.0
        res["n_gt"] = n_gt
        out[fld] = res
    return out


def receipt_lenient_match(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    """Share of receipts whose annotated fields are all correct under ``lenient_match``."""
    n = ok = 0
    for img_id, g in gts.items():
        present = [f for f in FIELDS if g.get(f)]
        if not present:
            continue
        n += 1
        p = preds.get(img_id) or {}
        if all(lenient_match(f, p.get(f), g[f]) for f in present):
            ok += 1
    return {"receipts": n, "all_fields_correct": ok, "rate": ok / n if n else 0.0}




# ---------------------------------------------------------------------------------- oracle, error causes, bootstrap
def oracle_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[str]]]:
    """Fields decoded from the *gold* word tags: the best any tagger can do with this OCR (OCR + mapping ceiling)."""
    return {rec["img_id"]: predict_fields(rec["words"], rec["labels"]) for rec in records}


ERROR_CAUSES = ("annotation_without_amount", "no_ocr_word_labelled", "ocr_text_error", "model_missed", "model_wrong")


def error_causes(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]],
                 preds: Dict[str, Dict[str, Optional[str]]], row_limit: int = 100) -> Dict[str, Any]:
    """Why each wrong field is wrong (lenient matching), one cause per (receipt, field).

    * ``annotation_without_amount``: TOTAL_COST region annotated but its text holds no number;
    * ``no_ocr_word_labelled``: no OCR word carries the field's gold tag (detection miss or label-mapping miss);
    * ``ocr_text_error``: the gold-tagged words exist but their OCR text does not match the annotation, so even a
      perfect tagger fails (for TOTAL_COST: the amount is not among the OCR amount candidates);
    * ``model_missed`` / ``model_wrong``: the OCR allowed a correct answer but the model gave none / a wrong one.
    """
    oracle = oracle_predictions(records)
    counts = {f: {c: 0 for c in ERROR_CAUSES} for f in FIELDS}
    rows: List[Dict[str, Any]] = []
    receipts_with_error = 0
    for rec, m in zip(records, merged):
        gt = rec.get("gt") or {}
        pred = preds.get(rec["img_id"]) or {}
        wrong_here = False
        for f in FIELDS:
            if not gt.get(f) or lenient_match(f, pred.get(f), gt[f]):
                continue
            wrong_here = True
            if f == "TOTAL_COST" and gt_amount(gt[f]) is None:
                cause = "annotation_without_amount"
            elif not any(t.endswith("-" + f) for t in rec["labels"]):
                cause = "no_ocr_word_labelled"
            elif f == "TOTAL_COST" and total_cause(rec, m["probs"]) == "amount_not_in_ocr":
                cause = "ocr_text_error"
            elif f != "TOTAL_COST" and not lenient_match(f, oracle[rec["img_id"]].get(f), gt[f]):
                cause = "ocr_text_error"
            elif not pred.get(f):
                cause = "model_missed"
            else:
                cause = "model_wrong"
            counts[f][cause] += 1
            if len(rows) < row_limit:
                rows.append({"file_id": rec.get("file_id"), "field": f, "cause": cause, "gold": " | ".join(gt[f]),
                             "predicted": pred.get(f), "oracle": oracle[rec["img_id"]].get(f)})
        receipts_with_error += wrong_here
    totals = {c: sum(counts[f][c] for f in FIELDS) for c in ERROR_CAUSES}
    return {"receipts": len(records), "receipts_with_error": receipts_with_error, "by_field": counts,
            "total": totals, "rows": rows}


def receipt_outcomes(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]]
                     ) -> Dict[str, Any]:
    """Per-receipt lenient outcome: ``[tp, fp, fn]`` for each field and whether all annotated fields are right."""
    ids = list(gts)
    out = {f: np.zeros((len(ids), 3), dtype=np.int64) for f in FIELDS}
    all_ok = np.zeros(len(ids), dtype=np.int64)
    has_any = np.zeros(len(ids), dtype=np.int64)
    for r, img_id in enumerate(ids):
        g, p = gts[img_id], preds.get(img_id) or {}
        ok_all = True
        for f in FIELDS:
            gt, pr = g.get(f), p.get(f)
            if pr and gt:
                hit = lenient_match(f, pr, gt)
                out[f][r] = (1, 0, 0) if hit else (0, 1, 1)
                ok_all &= hit
            elif pr and not gt:
                out[f][r] = (0, 1, 0)
            elif gt and not pr:
                out[f][r] = (0, 0, 1)
                ok_all = False
        has_any[r] = int(any(g.get(f) for f in FIELDS))
        all_ok[r] = int(ok_all and has_any[r])
    return {"ids": ids, "fields": out, "all_ok": all_ok, "has_any": has_any}


def _metrics_from_outcomes(oc: Dict[str, Any], idx: np.ndarray) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for f in FIELDS:
        tp, fp, fn = oc["fields"][f][idx].sum(axis=0)
        res[f] = prf1(int(tp), int(fp), int(fn))["f1"]
    res["macro_f1"] = float(np.mean([res[f] for f in FIELDS]))
    n = oc["has_any"][idx].sum()
    res["receipt_rate"] = float(oc["all_ok"][idx].sum() / n) if n else 0.0
    return res


def bootstrap_ci(oc: Dict[str, Any], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """95 % percentile intervals of field F1, macro F1 and receipt rate, resampling receipts with replacement."""
    n = len(oc["ids"])
    point = _metrics_from_outcomes(oc, np.arange(n))
    if n == 0:
        return {k: {"value": v, "low": v, "high": v} for k, v in point.items()}
    rng = np.random.default_rng(seed)
    samples = [_metrics_from_outcomes(oc, rng.integers(0, n, n)) for _ in range(n_boot)]
    out = {}
    for k, v in point.items():
        arr = np.array([s[k] for s in samples])
        out[k] = {"value": v, "low": float(np.quantile(arr, alpha / 2)), "high": float(np.quantile(arr, 1 - alpha / 2))}
    return out


def per_receipt(records: Sequence[Dict[str, Any]], preds: Dict[str, Dict[str, Optional[str]]],
                oc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Predictions and outcomes per receipt, so two systems can later be compared on the same receipts."""
    pos = {img_id: r for r, img_id in enumerate(oc["ids"])}
    rows = []
    for rec in records:
        r = pos[rec["img_id"]]
        rows.append({"img_id": rec["img_id"], "file_id": rec.get("file_id"), "pred": preds.get(rec["img_id"]),
                     "outcome": {f: oc["fields"][f][r].tolist() for f in FIELDS},
                     "all_ok": int(oc["all_ok"][r]), "has_any": int(oc["has_any"][r])})
    return rows


def outcomes_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Inverse of ``per_receipt`` (used by the comparison script)."""
    return {"ids": [r["img_id"] for r in rows],
            "fields": {f: np.array([r["outcome"][f] for r in rows], dtype=np.int64).reshape(-1, 3) for f in FIELDS},
            "all_ok": np.array([r["all_ok"] for r in rows], dtype=np.int64),
            "has_any": np.array([r["has_any"] for r in rows], dtype=np.int64)}


def paired_bootstrap(oc_a: Dict[str, Any], oc_b: Dict[str, Any], n_boot: int = 1000, seed: int = 0
                     ) -> Dict[str, Dict[str, float]]:
    """B minus A on the same receipts: point difference, 95 % interval and share of resamples where B > A."""
    pos_b = {img_id: r for r, img_id in enumerate(oc_b["ids"])}
    common = [i for i in oc_a["ids"] if i in pos_b]
    ia = np.array([oc_a["ids"].index(i) for i in common])
    ib = np.array([pos_b[i] for i in common])
    rng = np.random.default_rng(seed)
    n = len(common)
    base_a, base_b = _metrics_from_outcomes(oc_a, ia), _metrics_from_outcomes(oc_b, ib)
    diffs: Dict[str, List[float]] = {k: [] for k in base_a}
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        ma, mb = _metrics_from_outcomes(oc_a, ia[s]), _metrics_from_outcomes(oc_b, ib[s])
        for k in diffs:
            diffs[k].append(mb[k] - ma[k])
    metrics = {k: {"a": base_a[k], "b": base_b[k], "diff": base_b[k] - base_a[k],
                   "low": float(np.quantile(v, 0.025)), "high": float(np.quantile(v, 0.975)),
                   "p_b_better": float(np.mean(np.array(v) > 0))}
               for k, v in diffs.items()}
    return {"receipts": n, "metrics": metrics}


# ---------------------------------------------------------------------------------- report
def build_report(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]],
                 error_limit: int = 50, n_boot: int = 1000) -> Dict[str, Any]:
    """Everything the Week-4 evaluation needs from one set of predictions.

    ``fields`` / ``receipt_exact_match`` / ``total_amount`` use hard (argmax) decoding and strict text matching;
    the ``*_lenient_*`` and ``*_soft`` entries show how much is lost to OCR noise and to hard decoding;
    ``*_soft_rules`` additionally fills an empty TOTAL_COST / TIMESTAMP with the keyword / format rules
    (``src.extraction.rules``). ``fields_oracle`` decodes the gold tags: the ceiling set by OCR and label mapping.
    """
    preds = decode_predictions(records, merged)
    soft = decode_predictions_soft(records, merged)
    soft_rules = fill_missing(records, soft)
    gts = gold_fields(records)
    cm = word_confusion(records, merged)
    oc_soft, oc_rules = receipt_outcomes(soft, gts), receipt_outcomes(soft_rules, gts)
    return {
        "n_receipts": len(records),
        "fields": field_scores(preds, gts),
        "fields_lenient_hard": lenient_scores(preds, gts),
        "fields_lenient_soft": lenient_scores(soft, gts),
        "fields_lenient_soft_rules": lenient_scores(soft_rules, gts),
        "fields_oracle": lenient_scores(oracle_predictions(records), gts),
        "receipt_exact_match": receipt_exact_match(preds, gts),
        "receipt_exact_match_lenient_soft": receipt_lenient_match(soft, gts),
        "receipt_exact_match_lenient_soft_rules": receipt_lenient_match(soft_rules, gts),
        "bootstrap_soft": bootstrap_ci(oc_soft, n_boot),
        "bootstrap_soft_rules": bootstrap_ci(oc_rules, n_boot),
        "total_amount": amount_errors(preds, gts),
        "total_amount_soft": amount_errors(soft, gts),
        "total_amount_soft_rules": amount_errors(soft_rules, gts),
        "total_diagnostics": total_diagnostics(records, merged),
        "error_causes": error_causes(records, merged, soft_rules),
        "token_auc": token_auc(records, merged),
        "token_prf": token_prf(cm),
        "confusion_labels": list(LABELS),
        "confusion": cm.tolist(),
        "total_cost_errors": total_cost_errors(records, preds, error_limit),
        "per_receipt_soft": per_receipt(records, soft, oc_soft),
        "per_receipt_soft_rules": per_receipt(records, soft_rules, oc_rules),
    }


def _field_table(title: str, scores: Dict[str, Dict[str, float]]) -> List[str]:
    lines = [f"\n{title}", "field            P      R      F1     exact_match   n_gt"]
    for fld, s in scores.items():
        lines.append(f"{fld:<14} {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}   {s['exact_match']:.3f}"
                     f"         {int(s['n_gt'])}")
    return lines


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary of ``build_report`` (printed by the evaluation scripts)."""
    import json

    L: List[str] = []
    L += _field_table("STRICT, hard decoding", report["fields"])
    L += _field_table("LENIENT (OCR-tolerant), hard decoding", report["fields_lenient_hard"])
    L += _field_table("LENIENT, soft decoding for TOTAL_COST / TIMESTAMP  (model only)", report["fields_lenient_soft"])
    L += _field_table("LENIENT, soft decoding + rules fill empty TOTAL_COST / TIMESTAMP", report["fields_lenient_soft_rules"])
    L += _field_table("ORACLE: gold tags on the same OCR words (ceiling set by OCR + label mapping)", report["fields_oracle"])
    L.append("")
    L.append("receipt exact match, strict: " + json.dumps(report["receipt_exact_match"]))
    L.append("receipt exact match, lenient + soft: " + json.dumps(report["receipt_exact_match_lenient_soft"]))
    L.append("receipt exact match, lenient + soft + rules: " + json.dumps(report["receipt_exact_match_lenient_soft_rules"]))
    for key in ("bootstrap_soft", "bootstrap_soft_rules"):
        L.append(f"95% bootstrap CI ({key[10:]}):")
        for k, v in report[key].items():
            L.append(f"  {k:<14} {v['value']:.3f}  [{v['low']:.3f}, {v['high']:.3f}]")
    for key in ("total_amount", "total_amount_soft", "total_amount_soft_rules"):
        L.append(f"{key} (VND): " + json.dumps(report[key]))
    L.append("why the total is missing: " + json.dumps(report["total_diagnostics"]["categories"]))
    ec = report["error_causes"]
    L.append(f"error causes (final = soft + rules), {ec['receipts_with_error']}/{ec['receipts']} receipts with an error:")
    L.append("  field          " + "  ".join(f"{c[:14]:>14}" for c in ERROR_CAUSES))
    for f, row in ec["by_field"].items():
        L.append(f"  {f:<14} " + "  ".join(f"{row[c]:>14}" for c in ERROR_CAUSES))
    L.append(f"  {'ALL':<14} " + "  ".join(f"{ec['total'][c]:>14}" for c in ERROR_CAUSES))
    L.append("word precision / recall / F1 per label:")
    for lab, s in report["token_prf"].items():
        L.append(f"  {lab:<14} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  gold words={s['tp'] + s['fn']}")
    auc = report["token_auc"].get("macro")
    L.append(f"token ROC-AUC macro: {auc}")
    return "\n".join(L)
