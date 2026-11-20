"""Is the OCR of this receipt readable? Catches the text the fine-tuned VietOCR invents on unreadable images.

On images that the original VietOCR also could not read (it returned single letters), the fine-tuned VietOCR writes
phrases it learned in training instead: "Tổng thanh toán", "104 Trần Phú - Quảng Ninh", "10.000đ", again and again
(53 / 922 train, 5 / 115 validation receipts). Worse, the invented amounts can add up, so the item reader reports a
false ``verified`` (train ``...013crsgq``: items "10,000" = total "10,000", annotated total 79,800).

Signal: the share of words that belong to a three-word sequence occurring three or more times in the receipt. Normal
receipts stay low (99 % of train below 0.13; product lines repeat, but not three words three times); invented text is
high (median 0.73). The threshold 0.3 was chosen on train: it flags 48 of the 52 invented receipts and no other train
receipt. Pure Python.
"""
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPEAT_THRESHOLD = 0.3
MIN_WORDS = 15


def repeated_share(words: Sequence[str], n: int = 3, min_count: int = 3) -> float:
    """Share of words inside an ``n``-word sequence that occurs at least ``min_count`` times."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = set()
    for i, g in enumerate(grams):
        if counts[g] >= min_count:
            covered.update(range(i, i + n))
    return len(covered) / len(words)


def readability(words: Sequence[str], threshold: float = REPEAT_THRESHOLD) -> Dict[str, Any]:
    """``{"repeated_share", "unreadable"}``; too few words to judge counts as readable."""
    share = repeated_share(words)
    return {"repeated_share": round(share, 3), "unreadable": len(words) >= MIN_WORDS and share >= threshold}


def blank_unreadable(records: Sequence[Dict[str, Any]], preds: Dict[str, Dict[str, Optional[str]]]
                     ) -> Tuple[Dict[str, Dict[str, Optional[str]]], List[str]]:
    """Offline counterpart of ``ReceiptExtractor(quality_check=True)`` for scoring: a flagged receipt gets no prediction
    at all (the app answers "unreadable, take another photo"), so every annotated field of it counts as not read.

    Returns new predictions and the flagged ``img_id`` values; the input is not modified.
    """
    out = {k: dict(v) for k, v in preds.items()}
    flagged = [r["img_id"] for r in records if readability(r["words"])["unreadable"]]
    for img_id in flagged:
        out[img_id] = {}
    return out, flagged
