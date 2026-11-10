"""Field-level evaluation of token-classifier predictions (pure Python / NumPy / scikit-learn).

Entity-level F1 (``seqmetrics``) is a harsh training signal: one wrongly cut word makes a whole entity wrong.
What FinBud needs is the *value of each field*, so this module decodes the predicted tags into
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

from src.evaluation.metrics import FIELDS, cer, field_match, field_scores, normalize_amount, normalize_text, prf1
from src.extraction.decode import parse_amount_token, predict_fields
from src.preprocessing.token_dataset import ID2LABEL, LABEL2ID, LABELS

_TOTAL_IDS = [LABEL2ID["B-TOTAL_COST"], LABEL2ID["I-TOTAL_COST"]]
_TIMESTAMP_IDS = [LABEL2ID["B-TIMESTAMP"], LABEL2ID["I-TIMESTAMP"]]
_DATE_RE = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})")
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
    amounts = [normalize_amount(t) for t in (texts or [])]
    amounts = [a for a in amounts if a]
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
    m = _DATE_RE.search(text or "")
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
    for i, w in enumerate(words):
        parsed = parse_amount_token(w)
        if parsed is None:
            continue
        score = float(probs[i, _TOTAL_IDS].sum())
        if best is None or score >= best[0]:
            best = (score, parsed[0])
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
    """``(word index, value)`` of every amount-shaped word, rejoining ``52`` + ``700`` into 52700."""
    out = []
    for i, w in enumerate(words):
        parsed = parse_amount_token(w)
        if parsed is not None:
            out.append((i, parsed[1]))
        elif re.fullmatch(r"\d{1,3}", w.strip()) and i + 1 < len(words) and re.fullmatch(r"\d{3}", words[i + 1].strip()):
            value = int(w.strip() + words[i + 1].strip())
            if value >= 1000:
                out.append((i, value))
    return out


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
        truth = gt_amount((rec.get("gt") or {}).get("TOTAL_COST"))
        if truth is None:
            cats["no_gt_amount"] += 1
            continue
        scored = [(float(m["probs"][i, _TOTAL_IDS].sum()), i, v) for i, v in _amount_words(rec["words"])]
        match = [s for s in scored if s[2] == truth]
        if not match:
            cat = "amount_not_in_ocr"
        else:
            best_all = max(scored, key=lambda s: (s[0], s[1]))
            best_match = max(match, key=lambda s: (s[0], s[1]))
            if best_match[0] < min_prob:
                cat = "below_min_prob"
            elif best_match[:2] == best_all[:2]:
                cat = "top1"
            else:
                cat = "ranked_lower"
        cats[cat] += 1
        if len(examples[cat]) < example_limit:
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


def build_report(records: Sequence[Dict[str, Any]], merged: Sequence[Dict[str, Any]],
                 error_limit: int = 50) -> Dict[str, Any]:
    """Everything the Week-4 evaluation needs from one set of predictions.

    ``fields`` / ``receipt_exact_match`` / ``total_amount`` use hard (argmax) decoding and strict text matching;
    the ``*_lenient_*`` and ``*_soft`` entries show how much is lost to OCR noise and to hard decoding.
    """
    preds = decode_predictions(records, merged)
    soft = decode_predictions_soft(records, merged)
    gts = gold_fields(records)
    cm = word_confusion(records, merged)
    return {
        "n_receipts": len(records),
        "fields": field_scores(preds, gts),
        "fields_lenient_hard": lenient_scores(preds, gts),
        "fields_lenient_soft": lenient_scores(soft, gts),
        "receipt_exact_match": receipt_exact_match(preds, gts),
        "receipt_exact_match_lenient_soft": receipt_lenient_match(soft, gts),
        "total_amount": amount_errors(preds, gts),
        "total_amount_soft": amount_errors(soft, gts),
        "total_diagnostics": total_diagnostics(records, merged),
        "token_auc": token_auc(records, merged),
        "token_prf": token_prf(cm),
        "confusion_labels": list(LABELS),
        "confusion": cm.tolist(),
        "total_cost_errors": total_cost_errors(records, preds, error_limit),
    }
