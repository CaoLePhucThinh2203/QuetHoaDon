"""Does the end-to-end extractor give exactly the predictions that were scored in Week 4? (validation split, GPU)

Runs every validation receipt of the token dataset through ``ReceiptExtractor`` (the demo code path) with the given
checkpoint, then compares
1. the four predicted values per receipt with ``per_receipt_soft_rules`` stored in the Week-4 report, and
2. the lenient field F1, macro F1 and receipts-all-correct with the stored scores.

PASS means the demo reproduces the scored model. Small differences can only come from GPU non-determinism (fp16) and
are listed; anything larger is a bug to fix before reporting numbers. The test split is not read.

Example:
    python scripts/verify_extractor.py --checkpoint runs/m1b_v2_*/best --token-dir $TOK --image-dir $IMG \
        --reference eval/m1b_v2_val.json --out eval/verify_extractor_val.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm  # noqa: E402

from src.evaluation.reproduction import (compare_predictions, compare_scores, reference_predictions,  # noqa: E402
                                         score_predictions)
from src.extraction.schema import ANNOTATED_FIELDS  # noqa: E402
from src.pipeline.extractor import ReceiptExtractor  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the 'best' folder that produced --reference; globs allowed")
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--reference", required=True, help="Week-4 report of that checkpoint on validation (json)")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        sys.exit(f"No checkpoint matches {args.checkpoint}")
    checkpoint = matches[-1]
    with open(args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)
    if reference.get("split") not in (None, "validation"):
        sys.exit("--reference must be a validation report")
    ref_run = Path(str(reference.get("checkpoint", ""))).parent.name
    if ref_run and ref_run != Path(checkpoint).parent.name:
        print(f"[!] the reference was produced by run '{ref_run}', you passed '{Path(checkpoint).parent.name}'",
              file=sys.stderr)

    records = load_jsonl(str(Path(args.token_dir) / "validation.jsonl"))
    extractor = ReceiptExtractor.from_checkpoint(checkpoint, args.config_dir, with_ocr=False)
    preds, t0 = {}, time.time()
    for rec in tqdm(records, desc="verify"):
        out = extractor.extract_from_record(rec, args.image_dir)
        preds[rec["img_id"]] = {f: out["fields"][f]["text"] for f in ANNOTATED_FIELDS}
    seconds = time.time() - t0

    pred_cmp = compare_predictions(preds, reference_predictions(reference))
    score_cmp = compare_scores(score_predictions(records, preds), reference)
    passed = (score_cmp["identical"] and pred_cmp["receipts_identical"] == pred_cmp["receipts_compared"]
              and pred_cmp["receipts_compared"] == len(records))
    report = {"checkpoint": checkpoint, "reference": args.reference, "receipts": len(records),
              "seconds": round(seconds, 1), "predictions": pred_cmp, "scores": score_cmp, "pass": passed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"receipts compared {pred_cmp['receipts_compared']} / {len(records)}, identical {pred_cmp['receipts_identical']}")
    print("field mismatches:", pred_cmp["field_mismatches"])
    print("scores reference:", {k: round(v, 4) for k, v in score_cmp["reference"].items()})
    print("scores now      :", {k: round(v, 4) for k, v in score_cmp["new"].items()})
    for m in pred_cmp["examples"][:10]:
        print("  diff", m)
    print("PASS" if passed else "FAIL: the demo does not reproduce the scored predictions (see the examples above)")
    print("written:", args.out)


if __name__ == "__main__":
    main()
