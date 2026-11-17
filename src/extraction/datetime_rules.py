"""Read the date and the time of a receipt from OCR words (pure Python).

Why this exists: on the validation split only 16 of 80 predicted timestamps carried a time, although the OCR text of 32
of the other 64 receipts contained the annotated time. The Week-4 rule returns the date word alone, so the time is
dropped. This module keeps the date the Week-4 rule would pick and adds

* the time that follows the date (up to 4 words after it, or 2 words before it), or is glued to it;
* dates that OCR split into pieces: ``dd/mm`` followed by a year word (``20/09`` ``2019``), and the existing split
  ``14/08/2`` ``2020``.

The Week-4 decoding is not changed (the reported numbers stay reproducible); ``improve_timestamps`` is an optional
second step applied to final predictions.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.extraction.rules import DATE_SPLIT_RE
from src.training.field_eval import normalize_date

_DATE_ANY = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?:\s*[:.]\s*[0-5]\d)?(?!\d)", re.IGNORECASE)
_DAY_MONTH = re.compile(r"^\D*?(\d{1,2})[/\-.](\d{1,2})\D*$")
_YEAR_WORD = re.compile(r"^\D*(20[123]\d)\D*$")

TIME_WORDS_AFTER = 4
TIME_WORDS_BEFORE = 2


def extract_time(text: Optional[str]) -> Optional[str]:
    """``hh:mm`` of the first time-shaped part of ``text`` (dates are removed first), or ``None``."""
    s = DATE_SPLIT_RE.sub(" ", _DATE_ANY.sub(" ", text or ""))
    m = _TIME.search(s)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def _valid_day_month(day: int, month: int) -> bool:
    return 1 <= day <= 31 and 1 <= month <= 12


def date_at(words: Sequence[str], i: int) -> Optional[Tuple[str, int]]:
    """``(dd/mm/yyyy, index after the last word used)`` of a date that starts at word ``i``, or ``None``."""
    d = normalize_date(words[i])
    if d:
        return d, i + 1
    if i + 1 < len(words) and normalize_date(words[i + 1]) is None:
        d = normalize_date(words[i] + " " + words[i + 1])               # "14/08/2" + "2020"
        if d:
            return d, i + 2
    m = _DAY_MONTH.match(words[i])
    if m and _valid_day_month(int(m.group(1)), int(m.group(2))):
        for j in (i + 1, i + 2):                                        # "20/09" + "2019" (one filler word allowed)
            if j < len(words):
                y = _YEAR_WORD.match(words[j])
                if y:
                    return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{y.group(1)}", j + 1
    return None


def time_near(words: Sequence[str], start: int, end: int, allow_before: bool = True) -> Optional[str]:
    """Time glued to the date words, else the first time after them, else (if allowed) the first time just before them."""
    glued = extract_time(" ".join(words[start:end]))
    if glued:
        return glued
    after = extract_time(" ".join(words[end:end + TIME_WORDS_AFTER]))
    if after or not allow_before:
        return after
    return extract_time(" ".join(words[max(0, start - TIME_WORDS_BEFORE):start]))


def datetime_candidates(words: Sequence[str]) -> List[Dict[str, Any]]:
    """Every date in reading order with its time (or ``None``): ``{date, time, start, end}``.

    The "time just before the date" fallback is only used for the first date, so a later date never takes the time
    that belongs to the date before it.
    """
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(words):
        found = date_at(words, i)
        if found is None:
            i += 1
            continue
        date, end = found
        out.append({"date": date, "time": time_near(words, i, end, allow_before=not out), "start": i, "end": end})
        i = end
    return out


def read_datetime(words: Sequence[str]) -> Optional[str]:
    """``"dd/mm/yyyy hh:mm"`` (or the date alone) of the first date found, or ``None``."""
    cands = datetime_candidates(words)
    if not cands:
        return None
    first = cands[0]
    return f"{first['date']} {first['time']}" if first["time"] else first["date"]


def improve_timestamps(records: Sequence[Dict[str, Any]], preds: Dict[str, Dict[str, Optional[str]]]
                       ) -> Dict[str, Dict[str, Optional[str]]]:
    """Copy of ``preds`` whose ``TIMESTAMP`` is ``read_datetime`` of the receipt's words when that finds a date.

    Receipts without a readable date keep their current value, so a prediction is never made worse by a missing date.
    """
    out = {k: dict(v) for k, v in preds.items()}
    for rec in records:
        value = read_datetime(rec["words"])
        if value:
            out.setdefault(rec["img_id"], {})["TIMESTAMP"] = value
    return out
