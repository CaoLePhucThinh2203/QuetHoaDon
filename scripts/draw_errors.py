"""Draw receipts with wrong fields for manual review of the error analysis (CPU, Pillow).

For every receipt listed in the ``error_causes`` rows of an evaluation report it writes a PNG: the gold-labelled
words boxed (blue SELLER, green ADDRESS, orange TIMESTAMP, red TOTAL_COST) under a caption with, for each wrong
field, the automatic cause, the gold text and the final prediction. Open the images and check whether the cause
is right (OCR text wrong / label problem / model wrong).

Example:
    python scripts/draw_errors.py --report eval/m1b_v2_test.json --token-dir <token_dataset_v1> \
        --image-dir <preprocessed_images> --split test --out /kaggle/working/error_images --limit 30
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.evaluation.visualize import render_receipt  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=30, help="Number of receipts to draw")
    args = ap.parse_args()

    rows = json.load(open(args.report, encoding="utf-8"))["error_causes"]["rows"]
    by_receipt: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        by_receipt.setdefault(r["file_id"], []).append(r)
    records = {r["file_id"]: r for r in load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fid, errs in by_receipt.items():
        rec = records.get(fid)
        if rec is None:
            continue
        caption = [fid] + [f"{e['field']} [{e['cause']}]  gold: {e['gold']}  | pred: {e['predicted']}" for e in errs]
        with Image.open(Path(args.image_dir) / rec["image"]) as im:
            render_receipt(rec, im, caption).save(out / f"{n + 1:02d}_{fid}.png")
        n += 1
        if n >= args.limit:
            break
    print(f"drew {n} receipts into {out}")


if __name__ == "__main__":
    main()
