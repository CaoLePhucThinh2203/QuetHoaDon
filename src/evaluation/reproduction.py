"""Check that the demo extractor and today's code reproduce the scored Week-4 predictions (pure Python).

The Week-4 report of a checkpoint stores, for every validation receipt, the final predicted fields
(``per_receipt_soft_rules[*].pred``) and the field scores (``fields_lenient_soft_rules``). Running the same checkpoint
through ``ReceiptExtractor`` must give the same four values per receipt and therefore the same scores; any difference
means the demo does not do what was scored (or the code changed the decoding since then).
"""
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.metrics import FIELDS
from src.training.field_eval import gold_fields, lenient_scores, receipt_lenient_match

Preds = Dict[str, Dict[str, Optional[str]]]


def reference_predictions(report: Dict[str, Any]) -> Preds:
    """``{img_id: {field: text}}`` stored in an evaluation report (final decoding: soft + rules)."""
    return {row["img_id"]: dict(row["pred"] or {}) for row in report["per_receipt_soft_rules"]}


def compare_predictions(new: Preds, reference: Preds, fields: Sequence[str] = FIELDS) -> Dict[str, Any]:
    """Receipts and fields whose predicted text differs between ``new`` and ``reference`` (receipts in both)."""
    common = sorted(set(new) & set(reference))
    mismatches: List[Dict[str, Any]] = []
    per_field = {f: 0 for f in fields}
    for img_id in common:
        for f in fields:
            a, b = new[img_id].get(f), reference[img_id].get(f)
            if (a or None) != (b or None):
                per_field[f] += 1
                mismatches.append({"img_id": img_id, "field": f, "new": a, "reference": b})
    receipts_identical = len(common) - len({m["img_id"] for m in mismatches})
    return {"receipts_compared": len(common), "only_in_new": len(set(new) - set(reference)),
            "only_in_reference": len(set(reference) - set(new)), "receipts_identical": receipts_identical,
            "field_mismatches": per_field, "examples": mismatches[:20]}


def score_predictions(records: Sequence[Dict[str, Any]], preds: Preds) -> Dict[str, Any]:
    """Lenient field F1, macro F1 and receipts-all-correct, computed exactly as in the Week-4 report."""
    gts = gold_fields(records)
    fields = lenient_scores(preds, gts)
    macro = sum(fields[f]["f1"] for f in FIELDS) / len(FIELDS)
    return {"fields": {f: fields[f]["f1"] for f in FIELDS}, "macro_f1": macro,
            "receipts": receipt_lenient_match(preds, gts)}


def compare_scores(new: Dict[str, Any], reference_report: Dict[str, Any], tol: float = 1e-9) -> Dict[str, Any]:
    """Difference between freshly computed scores and the ones stored in the report."""
    ref_fields = {f: reference_report["fields_lenient_soft_rules"][f]["f1"] for f in FIELDS}
    ref_macro = sum(ref_fields.values()) / len(FIELDS)
    ref_ok = reference_report["receipt_exact_match_lenient_soft_rules"]["all_fields_correct"]
    diffs = {f: new["fields"][f] - ref_fields[f] for f in FIELDS}
    diffs["macro_f1"] = new["macro_f1"] - ref_macro
    same = all(abs(v) <= tol for v in diffs.values()) and new["receipts"]["all_fields_correct"] == ref_ok
    return {"reference": {**ref_fields, "macro_f1": ref_macro, "receipts_all_correct": ref_ok},
            "new": {**new["fields"], "macro_f1": new["macro_f1"],
                    "receipts_all_correct": new["receipts"]["all_fields_correct"]},
            "difference": diffs, "identical": same}
