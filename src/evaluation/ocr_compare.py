"""Compare OCR runs without any extraction model, through the token datasets built from them (pure Python).

Two views, on one split (validation for every decision):

* ``field_oracle``: for every annotated field, the OCR words labelled with that field (the words an ideal tagger would
  pick), joined in reading order, against the annotated text: exact match after ``strict_text`` and the character error
  rate. This is the ceiling a perfect LayoutXLM could reach on that OCR, so a better OCR run shows up here first.
* ``word_identity``: how many receipts of one run have exactly the same words as another run (used to check that a fresh
  run reproduces the OCR of the token dataset v1 before anything is compared with it).
"""
from typing import Any, Dict, List, Sequence

from src.evaluation.app_metrics import strict_text
from src.evaluation.metrics import cer
from src.extraction.schema import ANNOTATED_FIELDS


def field_words(record: Dict[str, Any], field: str) -> str:
    """Words tagged ``B-field`` / ``I-field`` in reading order, joined with spaces."""
    return " ".join(w for w, t in zip(record["words"], record["labels"]) if t != "O" and t.split("-", 1)[1] == field)


def field_oracle(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{field: {n, exact, exact_rate, mean_cer, covered}}`` over the receipts whose annotation has the field."""
    out: Dict[str, Dict[str, Any]] = {}
    for field in ANNOTATED_FIELDS:
        n = exact = covered = 0
        cers: List[float] = []
        for rec in records:
            gold_parts = (rec.get("gt") or {}).get(field)
            if not gold_parts:
                continue
            gold = strict_text(field, " ".join(gold_parts))
            if not gold:
                continue
            hyp = strict_text(field, field_words(rec, field))
            n += 1
            covered += bool(hyp)
            exact += gold == hyp
            cers.append(cer(gold, hyp))
        out[field] = {"n": n, "exact": exact, "exact_rate": exact / n if n else 0.0,
                      "mean_cer": sum(cers) / len(cers) if cers else 0.0, "covered": covered}
    return out


def word_identity(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Receipts present in both runs, how many have identical word lists, and the ids of the first differences."""
    words_b = {r["img_id"]: r["words"] for r in b}
    common = [r for r in a if r["img_id"] in words_b]
    differ = [r["img_id"] for r in common if r["words"] != words_b[r["img_id"]]]
    return {"only_in_a": len(a) - len(common), "only_in_b": len(words_b) - len(common), "common": len(common),
            "identical": len(common) - len(differ), "first_differences": differ[:10]}


def format_oracle_table(runs: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Markdown table: one row per run, exact / mean CER per field."""
    head = "| OCR run | " + " | ".join(f"{f} exact | {f} CER" for f in ANNOTATED_FIELDS) + " |"
    lines = [head, "|---|" + "---|---|" * len(ANNOTATED_FIELDS)]
    for name, o in runs.items():
        cells = []
        for f in ANNOTATED_FIELDS:
            s = o[f]
            cells.append(f"{s['exact']}/{s['n']} | {s['mean_cer']:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
