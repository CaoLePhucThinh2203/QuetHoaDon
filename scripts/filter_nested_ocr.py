"""Copy an OCR run with a stricter nested-box filter (CPU, seconds): drop a box when >= THR of it lies in a larger box.

Example:
    python scripts/filter_nested_ocr.py --in-dir $OCR/ocr_annotations --out-dir /kaggle/working/ocr_nest05/ocr_annotations --threshold 0.5
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.postprocess import filter_nested_items  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".json"))
    before = after = 0
    for name in files:
        with open(os.path.join(args.in_dir, name), encoding="utf-8") as f:
            meta = json.load(f)
        items = meta.get("ocr_items") or []
        kept = filter_nested_items(items, args.threshold)
        before, after = before + len(items), after + len(kept)
        meta["ocr_items"], meta["n_text_regions"] = kept, len(kept)
        meta["nested_threshold_offline"] = args.threshold
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    print(f"{len(files)} files | boxes {before} -> {after} (removed {before - after}) | written: {args.out_dir}")


if __name__ == "__main__":
    main()
