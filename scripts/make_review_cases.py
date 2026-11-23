"""Write review cases for reading receipts by eye (or by reviewer agents): photo + OCR lines + what the rules read.

Plan A (error review, TRAIN only): every train receipt whose item sum differs from the annotated total, plus a random
sample of correct ones as a control, one JSON per receipt in ``<out>/cases/``, split into chunk lists for parallel
reviewers. Plan C (web samples, VALIDATION): clear receipts whose items are verified and equal the annotated total,
from different stores, in ``<out>/web_candidates.json``. The test split is never read.

Each case holds: the original photo path, the annotated total, the OCR text grouped into lines, the item lines and
check the rules produced, the automatic failure kind and the unreadable-photo flag. No model is run.

Example:
    python scripts/make_review_cases.py --token-dir kaggle/phase1_eval/tok_ft_n05 \
        --images kaggle/mcocr/train_images/train_images --csv kaggle/mcocr/mcocr_train_df.csv --out kaggle/review --n 300
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import normalize_amount  # noqa: E402
from src.extraction.line_items import check_items, extract_items, group_lines  # noqa: E402
from src.recognition.quality import readability  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def gold_totals(rec: dict) -> set:
    return {a for a in (normalize_amount(t) for t in rec.get("gt", {}).get("TOTAL_COST", [])) if a and a < 100_000_000}


def failure_kind(read: dict, check: dict, gold: set, unreadable: bool) -> str:
    if unreadable:
        return "unreadable"
    if not read["items"]:
        return "no_items_no_header" if not read["header_found"] else "no_items_with_header"
    if any(not it["line_ok"] for it in read["items"]):
        return "line_arithmetic"
    return "sum_too_small" if check["sum"] < max(gold) else "sum_too_large"


def make_case(rec: dict, image_dir: str) -> dict:
    words, boxes = rec["words"], rec["boxes"]
    read = extract_items(words, boxes)
    check = check_items(read["items"], read["summary_amounts"])
    gold = gold_totals(rec)
    ok = bool(gold) and (check["sum"] in gold or check["sum_without_discounts"] in gold)
    quality = readability(words)
    return {
        "id": rec["file_id"], "image": os.path.abspath(os.path.join(image_dir, rec["img_id"])),
        "annotated_total": sorted(gold), "items_ok": ok,
        "auto_kind": "ok" if ok else (failure_kind(read, check, gold, quality["unreadable"]) if gold else "no_gold"),
        "unreadable_flag": quality["unreadable"],
        "ocr_lines": group_lines(words, boxes),
        "items_read": [{k: it[k] for k in ("name", "quantity", "unit_price", "amount", "line_ok")} for it in read["items"]],
        "check": {k: check[k] for k in ("sum", "status", "totals")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--images", required=True, help="folder with the ORIGINAL MC-OCR photos")
    ap.add_argument("--csv", required=True, help="mcocr_train_df.csv (image quality score)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="train cases: all failures, then correct ones up to n")
    ap.add_argument("--chunk", type=int, default=30, help="cases per reviewer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, "cases"), exist_ok=True)
    train = [make_case(r, args.images) for r in load_jsonl(os.path.join(args.token_dir, "train.jsonl"))]
    train = [c for c in train if c["annotated_total"] and os.path.exists(c["image"])]
    failed = [c for c in train if not c["items_ok"]]
    correct = [c for c in train if c["items_ok"]]
    rng.shuffle(correct)
    chosen = failed + correct[:max(0, args.n - len(failed))]
    rng.shuffle(chosen)
    for c in chosen:
        with open(os.path.join(args.out, "cases", c["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=1)
    chunks = [chosen[i:i + args.chunk] for i in range(0, len(chosen), args.chunk)]
    for k, ch in enumerate(chunks):
        with open(os.path.join(args.out, f"chunk_{k:02d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(os.path.abspath(os.path.join(args.out, "cases", c["id"] + ".json")) for c in ch) + "\n")

    quality = {r["img_id"]: float(r["anno_image_quality"]) for r in csv.DictReader(open(args.csv, encoding="utf-8"))}
    val = [make_case(r, args.images) for r in load_jsonl(os.path.join(args.token_dir, "validation.jsonl"))]
    cands = [c for c in val if c["items_ok"] and c["check"]["status"] == "verified" and len(c["items_read"]) >= 3
             and not c["unreadable_flag"] and os.path.exists(c["image"])]
    cands.sort(key=lambda c: -quality.get(os.path.basename(c["image"]), 0))
    seen, picked = set(), []
    for c in cands:                              # one receipt per store (first OCR line as the store name)
        store = (c["ocr_lines"] or [""])[0][:20].lower()
        if store in seen:
            continue
        seen.add(store)
        picked.append({**c, "image_quality": quality.get(os.path.basename(c["image"]))})
    with open(os.path.join(args.out, "web_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(picked[:12], f, ensure_ascii=False, indent=1)

    kinds = {}
    for c in chosen:
        kinds[c["auto_kind"]] = kinds.get(c["auto_kind"], 0) + 1
    print(json.dumps({"train_cases": len(chosen), "failed": len(failed), "control_correct": len(chosen) - len(failed),
                      "chunks": len(chunks), "auto_kinds": kinds, "web_candidates": len(picked[:12])},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
