"""Find near-duplicate receipts across train / validation / test (CPU; add --image-dir for the image signal).

Writes a JSON report with every flagged pair and, per evaluation split, how many receipts have a near-duplicate in
another split. Run it before trusting the scores; ``scripts/rescore_without_duplicates.py`` then scores the
test split without the flagged receipts.

Example:
    python scripts/find_duplicates.py --token-dir <token_dataset_v1> --image-dir <preprocessed_images> --out dup_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.duplicates import find_duplicates, leaking_receipts  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", default=None, help="Enables the image-hash signal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-threshold", type=float, default=0.8)
    ap.add_argument("--image-threshold", type=int, default=10)
    args = ap.parse_args()

    splits = {sp: load_jsonl(str(Path(args.token_dir) / f"{sp}.jsonl")) for sp in ("train", "validation", "test")}
    pairs = find_duplicates(splits, args.image_dir, args.text_threshold, image_threshold=args.image_threshold)
    cross = [p for p in pairs if p["split_a"] != p["split_b"]]
    within = len(pairs) - len(cross)
    leak = leaking_receipts(pairs)
    summary = {"pairs_total": len(pairs), "pairs_within_split": within, "pairs_across_splits": len(cross)}
    for sp, recs in splits.items():
        ids = {r["file_id"] for r in recs}
        summary[f"{sp}_receipts"] = len(recs)
        summary[f"{sp}_with_duplicate_in_other_split"] = sum(1 for i in ids if i in leak)
    by_signal = {}
    for p in cross:
        for s in p["signals"]:
            by_signal[s] = by_signal.get(s, 0) + 1
    summary["cross_pairs_by_signal"] = by_signal
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": pairs, "leaking": leak}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    for p in cross[:15]:
        print(p["split_a"], p["a"], "<->", p["split_b"], p["b"], p["signals"])
    print("full report:", args.out)


if __name__ == "__main__":
    main()
