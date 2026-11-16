"""Automatic first sort of OCR text errors, to speed up the manual review (pure Python).

The error-cause analysis already says that, for a field, the best the tagger could do (``oracle`` text, taken from the
gold-labelled words) still differs from the annotation. This module says *how* it differs, in a fixed order of rules:

``diacritics``     equal once Vietnamese diacritics are ignored
``spacing``        equal once all spaces are ignored (words split or glued)
``digits``         letters equal, digits differ
``truncated``      the OCR text is shorter than 70 % of the annotation (missing words or a cut line)
``extra_text``     the OCR text is longer than 140 % of the annotation
``other``          anything else (several kinds of errors at once)

The sort is a hint, not a verdict: the manual review confirms or corrects it.
"""
import re
from typing import Dict, Optional, Sequence

from src.evaluation.metrics import normalize_text

OCR_ERROR_TYPES = ("diacritics", "spacing", "digits", "truncated", "extra_text", "other")


def classify_ocr_error(gold: Optional[str], text: Optional[str]) -> str:
    """Error type of ``text`` against ``gold``; ``"none"`` when they are equal after normalisation."""
    g, t = normalize_text(gold), normalize_text(text)
    if g == t:
        return "none"
    gf, tf = normalize_text(gold, ignore_diacritics=True), normalize_text(text, ignore_diacritics=True)
    if gf == tf:
        return "diacritics"
    if gf.replace(" ", "") == tf.replace(" ", ""):
        return "spacing"
    if re.sub(r"\d", "", gf).replace(" ", "") == re.sub(r"\d", "", tf).replace(" ", ""):
        return "digits"
    ratio = len(tf.replace(" ", "")) / max(1, len(gf.replace(" ", "")))
    if ratio < 0.7:
        return "truncated"
    if ratio > 1.4:
        return "extra_text"
    return "other"


def count_types(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """``{field: {type: count}}`` for rows with ``field``, ``gold`` and ``oracle`` (the OCR text of the gold words)."""
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        kind = classify_ocr_error((r.get("gold") or "").replace("|", " "), r.get("oracle"))
        out.setdefault(r["field"], {k: 0 for k in OCR_ERROR_TYPES + ("none",)})[kind] += 1
    return out
