"""Near-duplicate receipts across the train / validation / test splits (pure Python, NumPy and Pillow).

The same receipt photographed twice (or two copies of one printed receipt) ends up in different splits and inflates
the scores. Three independent signals are used:

* ``gold``: same date + same total amount + same seller text (after normalisation);
* ``text``: Jaccard similarity of the sets of OCR words (>= ``text_threshold``);
* ``image``: Hamming distance of 256-bit difference hashes (<= ``image_threshold``), only when images are given.
"""
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import normalize_text
from src.training.field_eval import gt_amount, normalize_date


def gold_key(rec: Dict[str, Any]) -> Optional[Tuple[str, int, str]]:
    """``(date, total, seller)`` of a receipt, or ``None`` when any of the three is missing."""
    gt = rec.get("gt") or {}
    date = normalize_date(" ".join(gt.get("TIMESTAMP") or []))
    total = gt_amount(gt.get("TOTAL_COST"))
    seller = normalize_text(" ".join(gt.get("SELLER") or []), ignore_diacritics=True)
    if not date or total is None or not seller:
        return None
    return date, total, seller


def word_set(rec: Dict[str, Any], min_len: int = 3) -> frozenset:
    return frozenset(w for w in (normalize_text(x, ignore_diacritics=True) for x in rec["words"]) if len(w) >= min_len)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dhash(path: str, size: int = 16) -> np.ndarray:
    """``size * size`` bit difference hash (horizontal gradients of a small grey image)."""
    from PIL import Image

    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


def find_duplicates(splits: Dict[str, Sequence[Dict[str, Any]]], image_dir: Optional[str] = None,
                    text_threshold: float = 0.8, min_words: int = 15, image_threshold: int = 10) -> List[Dict[str, Any]]:
    """Every pair of receipts (any splits) flagged by at least one signal, with the signals that fired."""
    items = [(sp, rec) for sp, recs in splits.items() for rec in recs]
    keys = [gold_key(r) for _, r in items]
    sets = [word_set(r) for _, r in items]
    hashes = [None] * len(items)
    if image_dir:
        for i, (_, r) in enumerate(items):
            p = os.path.join(image_dir, r["image"])
            if os.path.exists(p):
                hashes[i] = dhash(p)
    pairs: List[Dict[str, Any]] = []
    for i, j in combinations(range(len(items)), 2):
        signals: Dict[str, Any] = {}
        if keys[i] is not None and keys[i] == keys[j]:
            signals["gold"] = True
        if len(sets[i]) >= min_words and len(sets[j]) >= min_words:
            jac = jaccard(sets[i], sets[j])
            if jac >= text_threshold:
                signals["text"] = round(jac, 3)
        if hashes[i] is not None and hashes[j] is not None:
            dist = int(np.count_nonzero(hashes[i] != hashes[j]))
            if dist <= image_threshold:
                signals["image"] = dist
        if signals:
            pairs.append({"a": items[i][1]["file_id"], "split_a": items[i][0], "b": items[j][1]["file_id"],
                          "split_b": items[j][0], "signals": signals})
    return pairs


def leaking_receipts(pairs: Sequence[Dict[str, Any]], min_signals: int = 1) -> Dict[str, List[str]]:
    """``{file_id: [file_ids of near-duplicates in another split]}`` for pairs that cross splits."""
    out: Dict[str, List[str]] = {}
    for p in pairs:
        if p["split_a"] == p["split_b"] or len(p["signals"]) < min_signals:
            continue
        out.setdefault(p["a"], []).append(p["b"])
        out.setdefault(p["b"], []).append(p["a"])
    return out
