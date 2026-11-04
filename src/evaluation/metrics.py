"""Evaluation metrics (pure Python, no model dependencies).

MC-OCR only annotates four fields - SELLER, ADDRESS, TIMESTAMP, TOTAL_COST - and only the
text regions that belong to them. Consequences for what can be measured:

* detection **recall** on annotated regions is valid; detection **precision** is not
  (unannotated text such as item lines would be counted as false positives);
* CER/WER are only valid on regions matched to an annotated region;
* ITEM_NAME / ITEM_PRICE / QUANTITY have no ground truth in MC-OCR.
"""
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.bbox_utils import compute_iou


# --------------------------------------------------------------------------- OCR errors
def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of one pair. Empty reference: 0.0 if hyp is empty else 1.0."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def corpus_error_rate(pairs: Iterable[Tuple[str, str]], level: str = "char") -> float:
    """Total edits / total reference length over many (ref, hyp) pairs (not a mean of ratios)."""
    edits = total = 0
    for ref, hyp in pairs:
        r, h = (ref, hyp) if level == "char" else (ref.split(), hyp.split())
        edits += edit_distance(r, h)
        total += len(r)
    return edits / total if total else 0.0


# --------------------------------------------------------------------------- P / R / F1
def prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- detection
def match_boxes(gt: List[List[float]], pred: List[List[float]],
                iou_thr: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. Boxes are xyxy. Returns (gt_i, pred_j, iou)."""
    cands = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = compute_iou(g, p)
            if iou >= iou_thr:
                cands.append((iou, i, j))
    cands.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in cands:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j, iou))
    return out


def detection_scores(gt: List[List[float]], pred: List[List[float]], iou_thr: float = 0.5) -> Dict[str, float]:
    """P/R/F1 at an IoU threshold. Read the module docstring before reporting ``precision``."""
    tp = len(match_boxes(gt, pred, iou_thr))
    return prf1(tp, fp=len(pred) - tp, fn=len(gt) - tp)


# --------------------------------------------------------------------------- normalisation
def normalize_text(s: Optional[str], ignore_diacritics: bool = False) -> str:
    """NFC, lower-case, collapse whitespace; optionally strip Vietnamese diacritics."""
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    if ignore_diacritics:
        s = unicodedata.normalize("NFD", s.replace("đ", "d"))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


_AMOUNT_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d{1,2})?")


def normalize_amount(text: Optional[str]) -> Optional[int]:
    """'27,000' / '27.000đ' / '100,000.00' / '1.250.000 VND' -> integer VND; None if no number."""
    if not text:
        return None
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = re.sub(r"(vnđ|vnd|đồng|₫|đ)", "", s)
    s = re.sub(r"[^0-9.,]", "", s)
    m = _AMOUNT_RE.fullmatch(s) if s else None
    if m:
        return int(re.sub(r"[.,]", "", m.group(1)))
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits else None


# --------------------------------------------------------------------------- field level
FIELDS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")


def group_gt_fields(instances: Iterable) -> Dict[str, List[str]]:
    """Group annotated region texts by label, top-to-bottom then left-to-right.

    ``instances`` are ``AnnotationInstance`` objects (``raw_bbox`` is xywh).
    """
    by_label: Dict[str, List[Tuple[float, float, str]]] = {}
    for inst in instances:
        x, y = inst.raw_bbox[0], inst.raw_bbox[1]
        by_label.setdefault(inst.label, []).append((y, x, inst.text))
    return {lab: [t for _, _, t in sorted(v)] for lab, v in by_label.items()}


def field_match(field: str, pred: Optional[str], gt_texts: Optional[List[str]],
                ignore_diacritics: bool = False) -> bool:
    """Exact match after normalisation.

    TOTAL_COST: predicted amount equals the amount of *any* annotated TOTAL_COST region.
    Other fields: normalised text equals the normalised, space-joined annotated regions.
    """
    if not pred or not gt_texts:
        return False
    if field == "TOTAL_COST":
        p = normalize_amount(pred)
        return p is not None and any(p == normalize_amount(g) for g in gt_texts)
    return normalize_text(pred, ignore_diacritics) == normalize_text(" ".join(gt_texts), ignore_diacritics)


def field_scores(preds: Dict[str, Dict[str, Optional[str]]], gts: Dict[str, Dict[str, List[str]]],
                 fields: Sequence[str] = FIELDS, ignore_diacritics: bool = False) -> Dict[str, Dict[str, float]]:
    """Per-field P/R/F1 + exact-match accuracy over receipts.

    preds[img_id][field] -> predicted text or None;  gts[img_id][field] -> annotated region texts.
    A wrong value counts as one FP *and* one FN; a missing prediction counts as FN;
    a prediction for a field without ground truth counts as FP.
    """
    out: Dict[str, Dict[str, float]] = {}
    for fld in fields:
        tp = fp = fn = n_gt = n_hit = 0
        for img_id, gt in gts.items():
            g = gt.get(fld)
            p = (preds.get(img_id) or {}).get(fld)
            if g:
                n_gt += 1
            if p and g:
                if field_match(fld, p, g, ignore_diacritics):
                    tp += 1
                    n_hit += 1
                else:
                    fp += 1
                    fn += 1
            elif p and not g:
                fp += 1
            elif g and not p:
                fn += 1
        res = prf1(tp, fp, fn)
        res["exact_match"] = n_hit / n_gt if n_gt else 0.0
        res["n_gt"] = n_gt
        out[fld] = res
    return out
