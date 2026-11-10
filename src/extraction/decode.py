"""Turn predicted BIO tags back into receipt fields (pure Python).

The token classifier tags OCR words; this module groups them into entities and picks one value per field in the
form the evaluation expects (``src.evaluation.metrics.field_scores``).

Policy (tune on validation, never on test)
* ``SELLER``: the best entity (highest mean score, then longest);
* ``ADDRESS`` / ``TIMESTAMP``: all entities in reading order, joined with a space (annotations are split by line);
* ``TOTAL_COST``: the amount token of the best entity. MC-OCR labels both "Tong tien:" and the number, so the
  entity usually contains a few words; only the numeric one is returned.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.metrics import normalize_amount
from src.extraction.schema import ANNOTATED_FIELDS

_SEP_AMOUNT = re.compile(r"^\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?$")
_PLAIN_AMOUNT = re.compile(r"^\d{4,9}$")
_CURRENCY_TAIL = re.compile(r"(vnđ|vnd|đồng|₫|đ|d)$")
_CURRENCY_HEAD = re.compile(r"^(vnđ|vnd|đ)")


def decode_entities(words: Sequence[str], tags: Sequence[str],
                    scores: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """Group consecutive tagged words into entities.

    ``I-X`` that does not continue an ``X`` entity starts a new one (lenient BIO decoding). Each entity has
    ``field``, ``start``, ``end`` (exclusive), ``words``, ``text`` and ``score`` (mean of ``scores``, or 1.0).
    """
    n = min(len(words), len(tags))
    entities: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for i in range(n):
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            cur = None
            continue
        prefix, field = tag.split("-", 1)
        if cur is not None and cur["field"] == field and prefix == "I":
            cur["end"] = i + 1
        else:
            cur = {"field": field, "start": i, "end": i + 1}
            entities.append(cur)
    for e in entities:
        ws = list(words[e["start"]:e["end"]])
        e["words"] = ws
        e["text"] = " ".join(ws)
        if scores is not None:
            seg = list(scores[e["start"]:e["end"]])
            e["score"] = sum(seg) / len(seg) if seg else 0.0
        else:
            e["score"] = 1.0
    return entities


def _clean_amount_token(token: str) -> str:
    t = token.strip().lower().strip("()[]:;*")
    t = _CURRENCY_HEAD.sub("", t)
    t = _CURRENCY_TAIL.sub("", t)
    return t.strip(".,")


# OCR (CRAFT boxes cut at the thousands separator) splits "107,000" into a head and a tail, e.g.
# "107,0" + ",000", "10,0" + "000", "253" + "000", "87,0" + "L000", "442,1" + "000", "97" + "000d".
_SPLIT_HEAD = re.compile(r"^(\d{1,3}(?:[.,]\d{3})*)(?:[.,]\d?)?$")
_SPLIT_TAIL = re.compile(r"^[.,]?(?:[l1i]000|\d{3})$")


def merge_split_amount(head: str, tail: str) -> Optional[int]:
    """Value in VND of an amount that OCR split into two words (``"107,0"`` + ``",000"`` -> 107000), else ``None``.

    The head keeps its complete digit groups (a stray digit after the last separator is the duplicated start of the
    tail and is dropped); the tail is ``000`` with an optional separator, a misread ``1``/``L`` and a currency sign.
    """
    ht, tt = _clean_amount_token(head), _clean_amount_token(tail)
    h = _SPLIT_HEAD.match(ht)
    if h is None or not _SPLIT_TAIL.match(tt):
        return None
    has_sep = bool(re.search(r"[.,]", head + tail))
    if tt.startswith("1") and len(tt) == 4 and not has_sep:
        return None                     # "253" + "1000" is more likely two numbers than 253,000
    if re.search(r"[.,]\d{3}", h.group(1)) and not tail.strip().startswith((",", ".")):
        return None                     # "12.000" + "000": the head is already a complete amount
    return int(re.sub(r"[.,]", "", h.group(1))) * 1000 + int(tt[-3:])


def amount_candidates(words: Sequence[str], min_value: int = 1000, max_value: int = 500_000_000
                      ) -> List[Tuple[int, int, str, int]]:
    """Every plausible amount among ``words`` as ``(start, end, text, value)`` (``end`` exclusive), in order.

    Single words follow ``parse_amount_token``; a head/tail pair split by OCR (``merge_split_amount``) is one
    candidate spanning both words and its words are not reused as single candidates.
    """
    out: List[Tuple[int, int, str, int]] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            merged = merge_split_amount(words[i], words[i + 1])
            if merged is not None and min_value <= merged <= max_value:
                out.append((i, i + 2, f"{merged:,}", merged))
                i += 2
                continue
        parsed = parse_amount_token(words[i], min_value, max_value)
        if parsed is not None:
            out.append((i, i + 1, parsed[0], parsed[1]))
        i += 1
    return out


def pick_amount(words: Sequence[str]) -> Optional[str]:
    """The money amount among ``words`` (the last one), or ``None``.

    Tokens with a thousands separator (``74,000``, ``1.250.000``) win over bare digit strings (``74000``), which
    could be codes. OCR often splits ``74,000`` into ``74`` and ``000`` (or ``74,0`` and ``,000``); such a pair is
    rejoined as ``74,000``.
    """
    separated: List[str] = []
    plain: List[str] = []
    i = 0
    while i < len(words):
        t = _clean_amount_token(words[i])
        merged = merge_split_amount(words[i], words[i + 1]) if i + 1 < len(words) else None
        if _SEP_AMOUNT.match(t):
            separated.append(t)
        elif merged is not None:
            separated.append(f"{merged:,}")
            i += 1
        elif _PLAIN_AMOUNT.match(t):
            plain.append(t)
        i += 1
    pool = separated or plain
    return pool[-1] if pool else None


def parse_amount_token(token: str, min_value: int = 1000, max_value: int = 500_000_000) -> Optional[Tuple[str, int]]:
    """``(cleaned text, value in VND)`` when ``token`` looks like a plausible receipt amount, else ``None``.

    Accepts ``74,000`` / ``1.250.000`` style tokens and bare 4-7 digit numbers; rejects phone numbers, codes and
    values outside ``[min_value, max_value]`` (these produced absurd totals when the model mislabelled them).
    """
    t = _clean_amount_token(token)
    if not (_SEP_AMOUNT.match(t) or re.fullmatch(r"\d{4,7}", t)):
        return None
    value = normalize_amount(t)
    if value is None or not (min_value <= value <= max_value):
        return None
    return t, value


def assemble_fields(entities: Sequence[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """One text value (or ``None``) for each annotated field."""
    out: Dict[str, Optional[str]] = {f: None for f in ANNOTATED_FIELDS}
    by_field: Dict[str, List[Dict[str, Any]]] = {f: [] for f in ANNOTATED_FIELDS}
    for e in entities:
        if e["field"] in by_field:
            by_field[e["field"]].append(e)

    if by_field["SELLER"]:
        best = max(by_field["SELLER"], key=lambda e: (e["score"], len(e["words"])))
        out["SELLER"] = best["text"]
    for fld in ("ADDRESS", "TIMESTAMP"):
        if by_field[fld]:
            out[fld] = " ".join(e["text"] for e in sorted(by_field[fld], key=lambda e: e["start"]))

    candidates = []
    for e in by_field["TOTAL_COST"]:
        amount = pick_amount(e["words"])
        if amount:
            candidates.append((e["score"], e["start"], amount))
    if candidates:
        out["TOTAL_COST"] = max(candidates)[2]
    return out


def predict_fields(words: Sequence[str], tags: Sequence[str],
                   scores: Optional[Sequence[float]] = None) -> Dict[str, Optional[str]]:
    """Words + predicted tags -> ``{field: text or None}``."""
    return assemble_fields(decode_entities(words, tags, scores))
