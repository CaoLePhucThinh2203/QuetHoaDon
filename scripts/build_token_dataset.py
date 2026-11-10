"""Build the BIO token dataset (train/validation/test .jsonl) from OCR JSON and MC-OCR annotations.

Example (Colab):
    python scripts/build_token_dataset.py \
        --csv   <dataset>/mcocr_train_df.csv \
        --ocr-dir pipeline_results/ocr_annotations \
        --split-manifest outputs/dataset/mcocr_manifest_split.csv \
        --out   outputs/token_dataset_v1

Read ``token_dataset_report.json`` before training: ``per_label.<field>.coverage`` is the share of annotated
regions that received at least one word, i.e. an upper bound for the recall of any model trained on these labels.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.token_dataset import DEFAULT_MIN_CONTAINMENT, build_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="MC-OCR annotation CSV (mcocr_train_df.csv)")
    ap.add_argument("--ocr-dir", required=True, help="Folder with <file_id>.json written by ReceiptPipeline")
    ap.add_argument("--split-manifest", required=True, help="mcocr_manifest_split.csv from prepare_dataset.py")
    ap.add_argument("--out", default="outputs/token_dataset", help="Output folder")
    ap.add_argument("--min-containment", type=float, default=DEFAULT_MIN_CONTAINMENT,
                    help="Share of a word box that must lie inside an annotation box to take its label")
    args = ap.parse_args()

    report = build_dataset(args.csv, args.ocr_dir, args.split_manifest, args.out, args.min_containment)
    summary = {
        "records_written": report["records_written"],
        "images_in_split_manifest": report["images_in_split_manifest"],
        "missing_ocr": len(report["missing_ocr"]),
        "missing_annotation": len(report["missing_annotation"]),
        "skipped": len(report["skipped"]),
        "images_by_status": report["all"]["images_by_status"],
        "per_label": report["all"]["per_label"],
        "words_per_image": report["all"]["words_per_image"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
