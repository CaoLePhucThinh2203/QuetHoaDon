"""Decode saved word probabilities, score them, and optionally check them against an evaluation report (CPU, no model).

``evaluate_layoutxlm.py --save-probs`` stores the word probabilities of a checkpoint. This script turns them into final
predictions with the same decoding as the scored results, so
* ``--reference`` (the report written in the same run) must be reproduced exactly: PASS, otherwise the stored probabilities
  or the decoding changed;
* decoding or rule changes can be scored in seconds, without GPU, before any model is retrained;
* the output holds the lenient (Week 4) scores, the strict app scores and the complete error-cause table
  (``error_causes.rows``, all rows) that ``draw_errors.py`` and ``ocr_error_catalog.py`` read.

Example:
    python scripts/decode_from_probs.py --token-dir $TOK --split validation --probs eval/m1b_v2_val_probs.npz \
        --reference eval/m1b_v2_val.json --out eval/decoded_val.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.app_metrics import app_scores, format_app_table  # noqa: E402
from src.evaluation.metrics import FIELDS  # noqa: E402
from src.evaluation.offline_decode import decode_from_probs  # noqa: E402
from src.evaluation.probs_io import load_probs  # noqa: E402
from src.extraction.datetime_rules import improve_timestamps  # noqa: E402
from src.extraction.field_cleanup import clean_predictions  # noqa: E402
from src.extraction.total_check import totals_from_items  # noqa: E402
from src.evaluation.reproduction import (compare_predictions, compare_scores, reference_predictions,  # noqa: E402
                                         score_predictions)
from src.training.data import load_jsonl  # noqa: E402
from src.training.field_eval import error_causes, gold_fields  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--probs", required=True, help=".npz written by evaluate_layoutxlm.py --save-probs")
    ap.add_argument("--reference", default=None, help="report of the same run; its predictions must be reproduced")
    ap.add_argument("--datetime", action="store_true",
                    help="add the time next to the date and rebuild split dates (src/extraction/datetime_rules.py)")
    ap.add_argument("--cleanup", action="store_true",
                    help="remove repeated words and glued labels / phone numbers from SELLER and ADDRESS")
    ap.add_argument("--total-from-items", action="store_true",
                    help="use the verified item sum when the total is not printed on a total line "
                         "(src/extraction/total_check.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.datetime or args.cleanup or args.total_from_items) and args.reference:
        sys.exit("--reference checks the Week-4 decoding; run it without --datetime / --cleanup / --total-from-items.")
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is only read for the final evaluation. Re-run with --allow-test.")

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    preds, merged = decode_from_probs(records, load_probs(args.probs))
    if args.datetime:
        preds = improve_timestamps(records, preds)
    if args.cleanup:
        preds = clean_predictions(preds)
    totals_changed = None
    if args.total_from_items:
        preds, totals_changed = totals_from_items(records, preds)
    gts = gold_fields(records)
    lenient = score_predictions(records, preds)
    app = app_scores(gts, preds)
    report = {"split": args.split, "receipts": len(records), "datetime_rules": args.datetime, "field_cleanup": args.cleanup,
              "total_from_items": args.total_from_items, "totals_changed": totals_changed, "lenient": lenient,
              "app": {k: v for k, v in app.items() if k != "per_receipt"},
              "error_causes": error_causes(records, merged, preds, row_limit=100000),
              "per_receipt_soft_rules": [{"img_id": r["img_id"], "file_id": r.get("file_id"), "pred": preds[r["img_id"]]}
                                         for r in records]}

    passed = None
    if args.reference:
        with open(args.reference, "r", encoding="utf-8") as f:
            ref = json.load(f)
        pred_cmp = compare_predictions(preds, reference_predictions(ref))
        score_cmp = compare_scores(lenient, ref)
        passed = (score_cmp["identical"] and pred_cmp["receipts_identical"] == pred_cmp["receipts_compared"]
                  and pred_cmp["receipts_compared"] == len(records))
        report["reproduction"] = {"predictions": pred_cmp, "scores": score_cmp, "pass": passed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"{args.split}: {len(records)} receipts | lenient macro F1 {lenient['macro_f1']:.4f} | "
          f"all correct {lenient['receipts']['all_fields_correct']}")
    print("lenient field F1:", {f: round(lenient["fields"][f], 4) for f in FIELDS})
    print(format_app_table(app, "Strict reading accuracy"))
    print("error causes:", report["error_causes"]["total"])
    if passed is not None:
        print("reproduction:", "PASS" if passed else "FAIL", "| field mismatches:",
              report["reproduction"]["predictions"]["field_mismatches"])
    print("written:", args.out)


if __name__ == "__main__":
    main()
