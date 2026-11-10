"""Error-analysis tables (Markdown) from one evaluation report (CPU only).

Reads the ``error_causes`` section that ``build_report`` computes for the final decoding (lenient, soft, rules)
and writes: causes per field with shares, and one row per wrong field (``--rows``, at least 50 for Week 4).

Example:
    python scripts/error_analysis.py --report eval/m1b_v2_val.json --out eval/error_analysis_m1b_val.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.field_eval import ERROR_CAUSES  # noqa: E402

MEANING = {
    "annotation_without_amount": "Annotation of the total holds no number",
    "no_ocr_word_labelled": "No OCR word carries the field (text detection or label mapping missed it)",
    "ocr_text_error": "OCR text of the field is wrong, even a perfect tagger fails",
    "model_missed": "OCR allowed the answer, the model predicted nothing",
    "model_wrong": "OCR allowed the answer, the model predicted a wrong value",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=100)
    args = ap.parse_args()
    with open(args.report, encoding="utf-8") as f:
        rep = json.load(f)
    ec = rep["error_causes"]
    total_errors = sum(ec["total"].values())
    L = [f"## Error analysis ({rep.get('split')}, {Path(args.report).name})", "",
         f"Receipts: {ec['receipts']}, with at least one wrong field: {ec['receipts_with_error']}. "
         f"Wrong fields: {total_errors}.", "",
         "| Cause | Meaning | " + " | ".join(ec["by_field"]) + " | All | Share |", "|" + "---|" * (len(ec["by_field"]) + 4)]
    for c in ERROR_CAUSES:
        share = ec["total"][c] / total_errors if total_errors else 0.0
        L.append(f"| {c} | {MEANING[c]} | " + " | ".join(str(ec["by_field"][f][c]) for f in ec["by_field"])
                 + f" | {ec['total'][c]} | {share:.0%} |")
    ocr_side = sum(ec["total"][c] for c in ERROR_CAUSES[:3])
    L += ["", f"OCR / annotation side: {ocr_side} ({ocr_side / max(1, total_errors):.0%}); "
              f"model side: {total_errors - ocr_side} ({(total_errors - ocr_side) / max(1, total_errors):.0%}).", "",
          "| # | Receipt | Field | Cause | Gold | Predicted | Best possible from OCR |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ec["rows"][:args.rows], 1):
        cells = [str(x if x is not None else "-").replace("|", "/")[:60] for x in
                 (r["file_id"], r["field"], r["cause"], r["gold"], r["predicted"], r["oracle"])]
        L.append(f"| {i} | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    print("written:", args.out)


if __name__ == "__main__":
    main()
