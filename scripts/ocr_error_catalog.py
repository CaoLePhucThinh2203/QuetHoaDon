"""Catalogue of OCR-text errors for the manual review (CPU, no model).

Takes the ``error_causes.rows`` of a report (use the output of ``decode_from_probs.py``, which keeps every row), keeps the
fields whose cause is ``ocr_text_error`` and sorts each into a first-guess error type (diacritics, spacing, digits,
truncated, extra text, other; see ``src/evaluation/ocr_errors.py``). It writes
* a Markdown summary with the counts per field and type, and
* a CSV with one row per wrong field and empty ``manual_type`` / ``note`` columns for the reviewer.

Example:
    python scripts/ocr_error_catalog.py --report eval/decoded_val.json --out-dir eval/ocr_errors
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ocr_errors import OCR_ERROR_TYPES, classify_ocr_error, count_types  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        causes = json.load(f)["error_causes"]
    rows = [r for r in causes["rows"] if r["cause"] == "ocr_text_error"]
    if len(causes["rows"]) < sum(causes["total"].values()):
        print("[!] the report keeps only part of the error rows; use the output of decode_from_probs.py", file=sys.stderr)
    counts = count_types(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ocr_errors_to_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "field", "auto_type", "gold", "ocr_text_of_gold_words", "final_prediction",
                    "manual_type", "note"])
        for r in rows:
            w.writerow([r["file_id"], r["field"], classify_ocr_error((r["gold"] or "").replace("|", " "), r["oracle"]),
                        r["gold"], r["oracle"], r["predicted"], "", ""])

    header = "| Field | Errors | " + " | ".join(OCR_ERROR_TYPES) + " |\n|---|---|" + "---|" * len(OCR_ERROR_TYPES) + "\n"
    lines = []
    for field, c in counts.items():
        n = sum(c[t] for t in OCR_ERROR_TYPES)
        lines.append(f"| {field} | {n} | " + " | ".join(str(c[t]) for t in OCR_ERROR_TYPES) + " |")
    text = header + "\n".join(lines) + "\n"
    (out / "ocr_errors_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"{len(rows)} OCR-text errors; review file: {out / 'ocr_errors_to_review.csv'}")


if __name__ == "__main__":
    main()
