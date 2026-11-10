"""Score a fine-tuned LayoutXLM checkpoint at field level (run on a GPU).

Reads one split of the token dataset, predicts every receipt, decodes the four fields and reports field P/R/F1,
exact match, the error of the total amount, ROC-AUC and the word confusion matrix.

The test split is scored ONCE, after every configuration is frozen: it needs ``--allow-test``.

Example (validation):
    python scripts/evaluate_layoutxlm.py --checkpoint runs/m1b_.../best \
        --token-dir <token_dataset_v1> --image-dir <preprocessed_images> --split validation --out m1b_val.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from src.training.data import load_jsonl  # noqa: E402
from src.training.engine import DEFAULTS, evaluate_records, load_model  # noqa: E402
from src.training.field_eval import build_report  # noqa: E402


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Folder written by save_pretrained (…/best)")
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true", help="Required to score the test split (do it once, at the end)")
    ap.add_argument("--out", required=True, help="JSON file for the full report")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--errors", type=int, default=50, help="How many wrong totals to list for the error analysis")
    ap.add_argument("--max-docs", type=int, default=None, help="Use only the first N receipts (smoke test)")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is scored once, after the configuration is frozen. Re-run with --allow-test.")

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    if args.max_docs:
        records = records[:args.max_docs]

    cfg = {**DEFAULTS, "batch_size": args.batch_size // 2 or 1, "num_workers": args.num_workers}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(cfg["model_name"], apply_ocr=False)
    model = load_model(args.checkpoint, False, device)

    t0 = time.time()
    ev = evaluate_records(model, records, processor, args.image_dir, cfg, device)
    seconds = time.time() - t0

    report = build_report(records, ev["predictions"], args.errors)
    report.update({
        "split": args.split, "checkpoint": args.checkpoint, "device": device,
        "entity_scores": ev["scores"], "loss": ev["loss"], "uncovered_words": ev["uncovered_words"],
        "n_words": ev["n_words"], "seconds_per_receipt": seconds / max(1, len(records)),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"split={args.split}  receipts={report['n_receipts']}  device={device}  "
          f"{report['seconds_per_receipt'] * 1000:.0f} ms / receipt (incl. image loading)")
    print("field            P      R      F1     exact_match   n_gt")
    for fld, s in report["fields"].items():
        print(f"{fld:<14} {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}   {s['exact_match']:.3f}         {int(s['n_gt'])}")
    for key, title in (("fields_lenient_hard", "LENIENT (OCR-tolerant), hard decoding"),
                       ("fields_lenient_soft", "LENIENT (OCR-tolerant), soft decoding for TOTAL_COST / TIMESTAMP")):
        print(f"\n{title}\nfield            P      R      F1     exact_match   n_gt")
        for fld, s in report[key].items():
            print(f"{fld:<14} {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}   {s['exact_match']:.3f}         {int(s['n_gt'])}")
    print("\nreceipt exact match, strict (all fields):", json.dumps(report["receipt_exact_match"]))
    print("receipt exact match, lenient + soft:", json.dumps(report["receipt_exact_match_lenient_soft"]))
    print("total amount errors (VND), hard:", json.dumps(report["total_amount"]))
    print("total amount errors (VND), soft:", json.dumps(report["total_amount_soft"]))
    print("why the total is missing (receipts with an annotated total):", json.dumps(report["total_diagnostics"]["categories"]))
    print("word precision / recall / F1 per label (word level):")
    for lab, s in report["token_prf"].items():
        print(f"  {lab:<14} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  gold words={s['tp'] + s['fn']}")
    print("entity micro F1:", round(report["entity_scores"]["micro"]["f1"], 4),
          "| token ROC-AUC macro:", report["token_auc"]["macro"])
    print("full report:", args.out)


if __name__ == "__main__":
    main()
