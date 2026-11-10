"""Keyword and format rules for receipt fields (pure Python).

Used two ways, always tuned on train / validation only:
* in the Week-3 baseline B0, where clusters alone cannot tell the total from any other number;
* as a fallback after any model: a field the model left empty is filled by the rule (``fill_missing``).

Rules
* ``TOTAL_COST``: the largest amount within a few words after a total keyword ("tổng cộng", "thanh toán",
  "total", ...); amounts split by OCR are rejoined (``amount_candidates``);
* ``TIMESTAMP``: the first date-shaped word (``dd/mm/yyyy``, ``dd-mm-yy``...).
"""
import re
import unicodedata
from typing import Dict, List, Optional, Sequence

from src.extraction.decode import amount_candidates

_DATE_WORD = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.](\d{4}|\d{2})")
# accent-free, lower-case keywords; matched on the concatenation of 1-3 consecutive words
_TOTAL_KEYS = ("tongcong", "tongtien", "tongthanhtoan", "thanhtoan", "tongso", "total", "phaitra",
               "canthanhtoan", "tongtt", "tongcongtien", "tong")
_EXCLUDE_KEYS = ("tienkhachdua", "khachdua", "tienthua", "thoilai", "tienmat", "chietkhau", "giamgia", "vat", "thue")


def fold(text: str) -> str:
    """Lower-case, no diacritics, only letters and digits (``"Tổng cộng:"`` -> ``"tongcong"``)."""
    s = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s)


def _keyword_at(words: Sequence[str], i: int, keys: Sequence[str]) -> bool:
    """True when the words starting at ``i`` (up to 3, folded and joined) begin with one of ``keys``."""
    if not fold(words[i]):
        return False
    joined = "".join(fold(w) for w in words[i:i + 3])
    return any(joined.startswith(k) for k in keys)


def rule_total(words: Sequence[str], window: int = 6) -> Optional[str]:
    """Largest amount found within ``window`` words after a total keyword (not after "tiền khách đưa"...)."""
    cands = amount_candidates(words)
    best = None
    for i in range(len(words)):
        if not _keyword_at(words, i, _TOTAL_KEYS) or _keyword_at(words, i, _EXCLUDE_KEYS):
            continue
        for start, stop, text, value in cands:
            if i < start <= i + window:
                if best is None or value > best[0]:
                    best = (value, text)
    return best[1] if best else None


def rule_timestamp(words: Sequence[str]) -> Optional[str]:
    """First date-shaped word, or ``None``."""
    for w in words:
        if _DATE_WORD.search(w):
            return w
    return None


def apply_rules(words: Sequence[str]) -> Dict[str, Optional[str]]:
    return {"TOTAL_COST": rule_total(words), "TIMESTAMP": rule_timestamp(words)}


PREFER_RULE = ("TIMESTAMP",)     # on validation the first date-shaped word beats the tagger (B0 .65 vs M1b .33)


def fill_missing(records: Sequence[Dict], preds: Dict[str, Dict[str, Optional[str]]],
                 prefer_rule: Sequence[str] = PREFER_RULE) -> Dict[str, Dict[str, Optional[str]]]:
    """Copy of ``preds`` where TOTAL_COST / TIMESTAMP come from the rules when the model gave none.

    Fields in ``prefer_rule`` use the rule value whenever the rule finds one, even if the model predicted something
    (chosen on validation only; the same decoding is applied to every system so comparisons stay fair).
    """
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for rec in records:
        p = dict(preds.get(rec["img_id"]) or {})
        for field, value in apply_rules(rec["words"]).items():
            if value and (not p.get(field) or field in prefer_rule):
                p[field] = value
        out[rec["img_id"]] = p
    return out


def rule_fields(records: Sequence[Dict]) -> Dict[str, Dict[str, Optional[str]]]:
    """Rules alone (no model): only TOTAL_COST and TIMESTAMP can be found this way."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for rec in records:
        out[rec["img_id"]] = {"SELLER": None, "ADDRESS": None, **apply_rules(rec["words"])}
    return out


__all__: List[str] = ["fold", "rule_total", "rule_timestamp", "apply_rules", "fill_missing", "rule_fields"]
