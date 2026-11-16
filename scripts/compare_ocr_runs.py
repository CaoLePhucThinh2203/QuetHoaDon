"""Ceiling of every OCR run on validation (perfect tagger) and word identity against a reference run (CPU).

Example:
    python scripts/compare_ocr_runs.py --reference v1=$TOK --run regen=/kaggle/working/tok_v1_regen \
        --run ft=/kaggle/working/tok_ft --out /kaggle/working/eval/ocr_runs_val.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ocr_compare import field_oracle, format_oracle_table, word_identity  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def _pair(text: str):
    name, path = text.split("=", 1)
    return name, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=_pair, help="name=token_dataset_dir of the reference run")
    ap.add_argument("--run", action="append", default=[], type=_pair, help="name=token_dataset_dir (repeatable)")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [args.reference] + args.run
    recs = {name: load_jsonl(os.path.join(path, f"{args.split}.jsonl")) for name, path in runs}
    oracle = {name: field_oracle(r) for name, r in recs.items()}
    ref_name = args.reference[0]
    identity = {name: word_identity(recs[name], recs[ref_name]) for name, _ in args.run}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "oracle": oracle, "identity_vs_reference": identity}, f, ensure_ascii=False, indent=2)
    print(f"OCR ceiling on {args.split} (words a perfect tagger would pick vs annotation, strict text):")
    print(format_oracle_table(oracle))
    for name, ident in identity.items():
        print(f"{name} vs {ref_name}: identical words in {ident['identical']}/{ident['common']} receipts "
              f"(only in {name}: {ident['only_in_a']}, only in {ref_name}: {ident['only_in_b']})")
    print("written:", args.out)


if __name__ == "__main__":
    main()
