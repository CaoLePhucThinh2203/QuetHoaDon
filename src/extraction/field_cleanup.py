"""Clean the SELLER and ADDRESS texts after decoding (pure Python, store-independent rules only).

Two kinds of OCR noise were seen on validation (fine-tuned OCR, M1b v3):

* repeated words from overlapping boxes: ``112 112``, ``p. p.cẩm``, ``q. q.pn``, ``co.op op``, ``k- k-park``;
* labels and numbers of the next line glued to the end: ``... quảng ninh 024``, ``... qn tel:``, ``... hà nội 0945984733``,
  ``... tong hop số``, ``... k-park ma``.

What is deliberately NOT done: snapping a prediction to a store name or address seen in the training data. 93 of the
106 validation sellers appear verbatim in train, so it would raise the MC-OCR score a lot while doing nothing for a store
the app has never seen. Every rule here looks only at the predicted text itself.
"""
import re
from typing import Dict, List, Optional

from src.extraction.rules import fold

# folded (lower-case, no diacritics, letters and digits only) words that start a new label, never end a name/address
_TRAILING_LABELS = {"tel", "dt", "dttel", "sdt", "dienthoai", "email", "mst", "ma", "maso", "masothue", "so", "tid",
                    "hotline", "hot", "terminal", "fax", "web", "website", "storeid", "tat", "el"}
_PHONE = re.compile(r"^[(+]?\d[\d.\-() ]{6,}\.?$")
_AREA_CODE = re.compile(r"^0\d{2,3}\.?$")


def _phone_like(tok: str) -> bool:
    return bool(_PHONE.match(tok)) and sum(c.isdigit() for c in tok) >= 8


def dedupe_tokens(text: Optional[str]) -> Optional[str]:
    """Remove a word repeated by overlapping OCR boxes.

    * the same word twice in a row (``112 112``, ``đường đường``);
    * a short piece ending in ``.`` or ``-`` right before a word that starts with it (``p. p.cẩm``, ``k- k-park``);
    * a short piece right after a word that ends with ``.`` + it (``co.op op``, ``co.op .op``).
    """
    if not text:
        return text
    toks = text.split()
    out: List[str] = []
    for i, t in enumerate(toks):
        low = t.lower()
        nxt = toks[i + 1].lower() if i + 1 < len(toks) else None
        if nxt is not None and low == nxt:
            continue
        if nxt is not None and len(low) <= 3 and low[-1] in ".-" and nxt.startswith(low) and len(nxt) > len(low):
            continue
        if out and len(low.lstrip(".")) <= 3 and low.lstrip(".") and out[-1].lower() != low \
                and out[-1].lower().endswith("." + low.lstrip(".")):
            continue
        out.append(t)
    return " ".join(out)


def strip_trailing_junk(text: Optional[str], field: str) -> Optional[str]:
    """Drop label words, phone numbers and (SELLER) short numbers glued to the end of the text."""
    if not text:
        return text
    toks = text.split()
    while len(toks) > 1:
        t = toks[-1]
        f = fold(t)
        if f in _TRAILING_LABELS or _phone_like(t) or (field == "ADDRESS" and _AREA_CODE.match(t)) \
                or (field == "SELLER" and re.fullmatch(r"\d{1,4}\.?", t)):
            toks.pop()
            continue
        break
    return " ".join(toks)


def clean_field(field: str, text: Optional[str]) -> Optional[str]:
    if field not in ("SELLER", "ADDRESS") or not text:
        return text
    return strip_trailing_junk(dedupe_tokens(text), field)


def clean_predictions(preds: Dict[str, Dict[str, Optional[str]]], fields=("SELLER", "ADDRESS")
                      ) -> Dict[str, Dict[str, Optional[str]]]:
    """Copy of ``preds`` with ``clean_field`` applied to ``fields``."""
    out = {k: dict(v) for k, v in preds.items()}
    for p in out.values():
        for f in fields:
            if f in p:
                p[f] = clean_field(f, p[f])
    return out
