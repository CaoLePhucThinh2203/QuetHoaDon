"""Does the offline scoring of the product configuration give exactly what the app returns? (CPU, seconds, no model)

The product configuration is the one the web app runs (``deploy/api/app.py``): the frozen final configuration plus the
item rules of 19/11, ``total_from_items=True`` and ``quality_check=True``. It is scored offline from saved word
probabilities with ``decode_from_probs.py --datetime --cleanup --total-from-items --quality-check`` and
``score_items.py --quality-check``. This script feeds the same probabilities to ``ReceiptExtractor`` (the app code, via
``extract_from_record``) and compares, receipt by receipt,

* the four fields of ``compact_result`` with the offline predictions (a receipt flagged unreadable must have the
  ``unreadable`` warning in the app and no prediction offline);
* the item lines with those of the item reader that ``score_items.py`` uses.

PASS means the numbers scored offline are the numbers of the app. It reads no annotation, so running it on the test split
scores nothing.

Example:
    python scripts/verify_product_path.py --token-dir phase1_eval/tok_ft_n05 --split validation \
        --probs eval/m1b_v3_val_probs.npz --out eval/verify_product_val.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.offline_decode import decode_from_probs  # noqa: E402
from src.evaluation.probs_io import load_probs  # noqa: E402
from src.extraction.datetime_rules import improve_timestamps  # noqa: E402
from src.extraction.field_cleanup import clean_predictions  # noqa: E402
from src.extraction.line_items import extract_items  # noqa: E402
from src.extraction.total_check import totals_from_items  # noqa: E402
from src.pipeline.extractor import ReceiptExtractor, compact_result  # noqa: E402
from src.recognition.quality import blank_unreadable  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402

FIELDS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")


def offline_predictions(records, probs):
    preds, _ = decode_from_probs(records, probs)
    preds = clean_predictions(improve_timestamps(records, preds))
    preds, _ = totals_from_items(records, preds)
    return blank_unreadable(records, preds)


def compare(records, probs):
    offline, flagged = offline_predictions(records, probs)
    flagged = set(flagged)
    ex = ReceiptExtractor(lambda rec, image_dir: probs[rec["img_id"]], postprocess=True, items=True,
                          total_from_items=True, quality_check=True)
    rows = []
    for rec in records:
        app = compact_result(ex.extract_from_record(rec, "."))
        img = rec["img_id"]
        app_fields = {f: (app["fields"].get(f) or {}).get("text") for f in FIELDS}
        off_fields = {f: offline[img].get(f) or None for f in FIELDS}
        app_flag = app.get("warning") == "unreadable"
        items_app = [{k: it[k] for k in ("name", "quantity", "unit_price", "amount")} for it in app.get("items", [])]
        items_off = [] if img in flagged else [{k: it[k] for k in ("name", "quantity", "unit_price", "amount")}
                                               for it in extract_items(rec["words"], rec["boxes"])["items"]]
        diff = [f for f in FIELDS if app_fields[f] != off_fields[f]]
        rows.append({"img_id": img, "flag_app": app_flag, "flag_offline": img in flagged, "field_diffs": diff,
                     "items_identical": items_app == items_off,
                     "identical": not diff and app_flag == (img in flagged) and items_app == items_off})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("Re-run with --allow-test (this check reads no annotation, but the flag keeps test runs explicit).")

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    rows = compare(records, load_probs(args.probs))
    n_same = sum(r["identical"] for r in rows)
    field_diffs = {f: sum(f in r["field_diffs"] for r in rows) for f in FIELDS}
    passed = n_same == len(rows) == len(records)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "receipts": len(rows), "identical": n_same, "field_diffs": field_diffs,
                   "flagged": sum(r["flag_offline"] for r in rows), "pass": passed,
                   "differences": [r for r in rows if not r["identical"]]}, f, ensure_ascii=False, indent=2)
    print(f"receipts compared {len(rows)} / {len(records)}, identical {n_same}")
    print("field differences:", field_diffs, "| item differences:", sum(not r["items_identical"] for r in rows),
          "| flagged unreadable:", sum(r["flag_offline"] for r in rows))
    print("PASS" if passed else "FAIL")
    print("written:", args.out)


if __name__ == "__main__":
    main()
