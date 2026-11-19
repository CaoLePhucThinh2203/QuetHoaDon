"""Cross-check of the predicted total with the item lines.

The item reader (``line_items.py``) marks a receipt ``verified`` when the item amounts add up to an amount printed on one
of the receipt's own "total" lines. On train (781 receipts with an annotated total, item rules v3) such a sum equals the
annotated total on 530 of 547 verified receipts (96.9 %), so it is a strong second opinion on the total:

* the model's total is **kept** when it is one of the amounts printed on a "total" line (that also keeps a total after
  an order-level discount, "Tổng 78 000 -> chiết khấu -> 70 200", where the item sum is the amount before the discount);
* otherwise, or when the model found no total, the verified item sum is used.

There is no parameter to tune. Pure Python.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.metrics import normalize_amount
from src.extraction.line_items import check_items, extract_items


def choose_total(predicted: Optional[int], check: Dict[str, Any]) -> Optional[int]:
    """The amount to use instead of the model's total, or ``None`` to keep the model's total."""
    if check.get("status") != "verified":
        return None
    printed = set(check.get("totals") or [])
    item_sum = check["sum"] if check["sum"] in printed else check["sum_without_discounts"]
    if predicted is not None and (predicted in printed or predicted == item_sum):
        return None
    return item_sum


def format_amount(value: int) -> str:
    """``585500`` -> ``"585,500"`` (the style of the receipts; ``normalize_amount`` reads it back)."""
    return f"{value:,}"


def totals_from_items(records: List[Dict[str, Any]], preds: Dict[str, Dict[str, Any]]
                      ) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Apply ``choose_total`` to decoded predictions (``img_id -> {field: text}``); returns new predictions and the
    number of receipts whose total changed. The input is not modified."""
    out = {k: dict(v) for k, v in preds.items()}
    changed = 0
    for rec in records:
        pred = out.setdefault(rec["img_id"], {})
        text = pred.get("TOTAL_COST")
        read = extract_items(rec["words"], rec["boxes"])
        new = choose_total(normalize_amount(text) if text else None,
                           check_items(read["items"], read["summary_amounts"]))
        if new is not None:
            pred["TOTAL_COST"] = format_amount(new)
            changed += 1
    return out, changed
