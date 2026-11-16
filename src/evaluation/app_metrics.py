"""Reading accuracy from the point of view of an app that shows the extracted values to a user (pure Python).

The Week-4 scores are lenient on purpose (a seller or address counts as right when up to 30 % of its characters are
wrong, a timestamp when the calendar date matches). A user sees the text, so these metrics are strict:

* ``TOTAL_COST``: the predicted amount equals an annotated amount (only receipts whose annotation holds a plausible amount);
* ``TIMESTAMP``: the date equals the annotated date; the time ``hh:mm`` is scored separately where the annotation has one;
* ``SELLER`` / ``ADDRESS``: the text equals the annotated text after normalisation (case, spaces, and for addresses the
  prefix "Địa chỉ:" / "ĐC:"); the mean character error rate is reported next to it;
* a receipt is "all correct" when every field present in its annotation is right by the rules above.

Accuracy intervals are Wilson score intervals (95 %). Everything works on ``{img_id: {field: text}}`` dictionaries, the
same form as the evaluation reports and ``src.training.field_eval.gold_fields``.
"""
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.metrics import cer, normalize_amount, normalize_text
from src.extraction.rules import DATE_SPLIT_RE
from src.training.field_eval import PLAUSIBLE_MAX, PLAUSIBLE_MIN, gold_amounts, gt_amount, normalize_date

APP_FIELDS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")

_DATE_ANY = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?:\s*[:.]\s*[0-5]\d)?(?!\d)", re.IGNORECASE)
_ADDRESS_PREFIX = re.compile(r"^\s*(?:địa\s*chỉ|dia\s*chi|đ\s*/\s*c|đ\.\s*c|đc|add(?:ress)?)\s*[:.\-]?\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------------- normalisation
def strict_text(field: str, text: Optional[str]) -> str:
    """Normalised text used for the strict comparison of SELLER / ADDRESS."""
    s = unicodedata.normalize("NFC", text or "")
    if field == "ADDRESS":
        s = _ADDRESS_PREFIX.sub("", s)
    return normalize_text(s)


def extract_time(text: Optional[str]) -> Optional[str]:
    """``hh:mm`` of the first time-shaped part of ``text`` (dates are removed first), or ``None``."""
    s = DATE_SPLIT_RE.sub(" ", _DATE_ANY.sub(" ", text or ""))
    m = _TIME.search(s)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % Wilson score interval of a proportion ``k / n`` (``(0, 0)`` when ``n`` is 0)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _stat(correct: int, n: int, **extra: Any) -> Dict[str, Any]:
    low, high = wilson_interval(correct, n)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0, "ci_low": low, "ci_high": high, **extra}


# ---------------------------------------------------------------------------------- scoring
def _joined(texts: Optional[Sequence[str]]) -> str:
    return " ".join(texts or [])


def app_scores(gts: Dict[str, Dict[str, List[str]]], preds: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    """Strict reading accuracy of ``preds`` against the annotations ``gts`` (receipts of ``gts`` only)."""
    flags: Dict[str, Dict[str, bool]] = {}
    total = {"n": 0, "correct": 0, "none": 0}
    date = {"n": 0, "correct": 0, "none": 0}
    clock = {"n": 0, "correct": 0}
    text = {f: {"n": 0, "correct": 0, "none": 0, "cer": []} for f in ("SELLER", "ADDRESS")}

    for img_id, gt in gts.items():
        pred = preds.get(img_id) or {}
        ok: Dict[str, bool] = {}

        gold_total = gt_amount(gt.get("TOTAL_COST"))
        if gold_total is not None:
            plausible = {a for a in gold_amounts(gt.get("TOTAL_COST")) if PLAUSIBLE_MIN <= a <= PLAUSIBLE_MAX}
            amount = normalize_amount(pred.get("TOTAL_COST")) if pred.get("TOTAL_COST") else None
            total["n"] += 1
            total["none"] += amount is None
            ok["TOTAL_COST"] = amount is not None and amount in plausible
            total["correct"] += ok["TOTAL_COST"]

        gold_stamp = _joined(gt.get("TIMESTAMP"))
        gold_date = normalize_date(gold_stamp)
        if gold_date is not None:
            gold_time = extract_time(gold_stamp)
            pred_stamp = pred.get("TIMESTAMP") or ""
            date["n"] += 1
            date["none"] += normalize_date(pred_stamp) is None
            date_ok = normalize_date(pred_stamp) == gold_date
            date["correct"] += date_ok
            ok["TIMESTAMP"] = date_ok
            if gold_time is not None:
                clock["n"] += 1
                time_ok = extract_time(pred_stamp) == gold_time
                clock["correct"] += time_ok
                ok["TIMESTAMP"] = date_ok and time_ok

        for field in ("SELLER", "ADDRESS"):
            if not gt.get(field):
                continue
            gold_text, pred_text = strict_text(field, _joined(gt[field])), strict_text(field, pred.get(field))
            s = text[field]
            s["n"] += 1
            s["none"] += not pred_text
            s["cer"].append(cer(gold_text, pred_text) if gold_text else 0.0)
            ok[field] = bool(gold_text) and gold_text == pred_text
            s["correct"] += ok[field]

        flags[img_id] = ok

    scored = [ok for ok in flags.values() if ok]
    all_correct = sum(1 for ok in scored if all(ok.values()))
    fields = {
        "SELLER": _stat(text["SELLER"]["correct"], text["SELLER"]["n"], no_prediction=text["SELLER"]["none"],
                        mean_cer=_mean(text["SELLER"]["cer"])),
        "ADDRESS": _stat(text["ADDRESS"]["correct"], text["ADDRESS"]["n"], no_prediction=text["ADDRESS"]["none"],
                         mean_cer=_mean(text["ADDRESS"]["cer"])),
        "TIMESTAMP": _stat(date["correct"], date["n"], no_prediction=date["none"], time=_stat(clock["correct"], clock["n"])),
        "TOTAL_COST": _stat(total["correct"], total["n"], no_prediction=total["none"]),
    }
    return {"fields": fields, "receipts_all_correct": _stat(all_correct, len(scored)), "per_receipt": flags}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_app_table(scores: Dict[str, Any], title: str = "") -> str:
    lines = [f"### {title}" if title else "", "| Field | Correct | Accuracy | 95 % interval | No prediction | Mean CER |",
             "|---|---|---|---|---|---|"]
    for name in APP_FIELDS:
        f = scores["fields"][name]
        cer_cell = f"{f['mean_cer']:.3f}" if "mean_cer" in f else "-"
        lines.append(f"| {name} | {f['correct']} / {f['n']} | {100 * f['accuracy']:.1f} % | "
                     f"[{100 * f['ci_low']:.1f}, {100 * f['ci_high']:.1f}] | {f['no_prediction']} | {cer_cell} |")
    t = scores["fields"]["TIMESTAMP"]["time"]
    lines.append(f"| TIMESTAMP time hh:mm | {t['correct']} / {t['n']} | {100 * t['accuracy']:.1f} % | "
                 f"[{100 * t['ci_low']:.1f}, {100 * t['ci_high']:.1f}] | - | - |")
    a = scores["receipts_all_correct"]
    lines.append(f"| **All fields correct** | {a['correct']} / {a['n']} | {100 * a['accuracy']:.1f} % | "
                 f"[{100 * a['ci_low']:.1f}, {100 * a['ci_high']:.1f}] | - | - |")
    return "\n".join(l for l in lines if l is not None) + "\n"
