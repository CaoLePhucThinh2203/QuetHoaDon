"""Create manifests and invoice-level train/validation/test splits for MC-OCR.

Example:
    python scripts/prepare_dataset.py \
        --csv  <dataset>/mcocr_train_df.csv \
        --images <dataset>/train_images/train_images \
        --out  outputs/dataset
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.dataset_preparer import prepare_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="MC-OCR annotation CSV (mcocr_train_df.csv)")
    ap.add_argument("--images", required=True, help="Folder with the receipt images")
    ap.add_argument("--out", default="outputs/dataset", help="Where manifests + dataset_summary.json go")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = prepare_dataset(args.csv, args.images, args.out, random_seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
