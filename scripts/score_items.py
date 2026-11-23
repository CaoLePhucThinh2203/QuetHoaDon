"""Score the item reader (``src/extraction/line_items.py``) with the annotated total instead of item labels.

MC-OCR has no item labels, so the check is arithmetic. For every receipt that has an annotated ``TOTAL_COST``:

* ``found``      at least one item line was read;
* ``sum_gold``   the item amounts add up to the annotated total (items without discount lines, or with them);
* ``verified``   the receipt's own printed total line equals that sum (what the app can know without labels);
* precision of ``verified``: of the verified receipts, how many really equal the annotated total.

Item NAMES cannot be scored this way; the report keeps a few random receipts for reading by eye.
The test split needs ``--allow-test`` and must be scored once, after the rules are frozen.

Example:
    python scripts/score_items.py --token-dir phase1_eval/tok_ft_n05 --split train --out eval/items_train.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.app_metrics import wilson_interval  # noqa: E402
from src.evaluation.metrics import normalize_amount  # noqa: E402
from src.extraction.line_items import check_items, extract_items  # noqa: E402
from src.recognition.quality import readability  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def gold_amounts(gt: dict) -> set:
    """Annotated total amounts; a value of 100 million or more is two amounts glued together by the annotation."""
    return {a for a in (normalize_amount(t) for t in gt.get("TOTAL_COST", [])) if a and a < 100_000_000}


def score_records(records: list, n_examples: int = 15, seed: int = 0, quality_check: bool = False) -> dict:
    """With ``quality_check`` a receipt flagged unreadable shows no item (as in the app): not found, not verified."""
    rows = []
    flagged = 0
    for rec in records:
        gold = gold_amounts(rec.get("gt", {}))
        if not gold:
            continue
        res = extract_items(rec["words"], rec["boxes"])
        chk = check_items(res["items"], res["summary_amounts"])
        if quality_check and readability(rec["words"])["unreadable"]:
            flagged += 1
            res = dict(res, items=[])
            chk = dict(chk, status="unreadable", sum=0, sum_without_discounts=0, lines_ok=0)
        match = bool(res["items"]) and (chk["sum"] in gold or chk["sum_without_discounts"] in gold)
        rows.append({"img_id": rec["img_id"], "gold": sorted(gold), "n_items": len(res["items"]), "sum": chk["sum"],
                     "header": res["header_found"], "end": res["end_found"], "lines_ok": chk["lines_ok"],
                     "verified": chk["status"] == "verified", "match": match, "items": res["items"],
                     "summary_amounts": res["summary_amounts"]})
    n = len(rows)
    found = [r for r in rows if r["n_items"]]
    ver = [r for r in rows if r["verified"]]
    ver_ok = [r for r in ver if r["match"]]
    n_items = sum(r["n_items"] for r in rows)
    stat = lambda k, m: {"count": k, "of": m, "rate": k / m if m else None, "ci95": wilson_interval(k, m) if m else None}
    random.seed(seed)
    examples = random.sample(found, min(n_examples, len(found)))
    return {
        "receipts_with_gold_total": n,
        "quality_check": quality_check,
        "flagged_unreadable": flagged,
        "found_items": stat(len(found), n),
        "sum_matches_gold_total": stat(sum(r["match"] for r in rows), n),
        "receipts_with_2plus_items": stat(sum(1 for r in rows if r["n_items"] >= 2), n),
        "sum_matches_with_2plus_items": stat(sum(1 for r in rows if r["n_items"] >= 2 and r["match"]),
                                             sum(1 for r in rows if r["n_items"] >= 2)),
        "verified": stat(len(ver), n),
        "verified_and_matches_gold": stat(len(ver_ok), len(ver)),
        "header_found": stat(sum(r["header"] for r in rows), n),
        "end_found": stat(sum(r["end"] for r in rows), n),
        "items_total": n_items,
        "lines_arithmetic_ok": stat(sum(r["lines_ok"] for r in rows), n_items),
        "examples": examples,
        "failures": [r for r in found if not r["match"]][:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--quality-check", action="store_true",
                    help="receipts flagged unreadable (src/recognition/quality.py) show no item, as in the app")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is scored once, after the rules are frozen. Re-run with --allow-test.")

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    report = score_records(records, quality_check=args.quality_check)
    report["split"] = args.split
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"### Item reading, {args.split} ({report['receipts_with_gold_total']} receipts with an annotated total)")
    print("| Check | Count | Rate | 95 % interval |\n|---|---|---|---|")
    for key, label in (("found_items", "at least one item read"), ("sum_matches_gold_total", "item sum = annotated total"),
                       ("receipts_with_2plus_items", "two or more items read"),
                       ("sum_matches_with_2plus_items", "of those, item sum = annotated total"),
                       ("verified", "verified (sum = printed total line)"),
                       ("verified_and_matches_gold", "of verified, sum = annotated total"),
                       ("header_found", "column header found"), ("end_found", "total line found"),
                       ("lines_arithmetic_ok", "item lines with qty x price = amount")):
        s = report[key]
        ci = f"[{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}]" if s["ci95"] else "-"
        print(f"| {label} | {s['count']} / {s['of']} | {100 * s['rate']:.1f} % | {ci} |" if s["rate"] is not None
              else f"| {label} | 0 / 0 | - | - |")
    if args.quality_check:
        print("flagged unreadable (no items shown):", report["flagged_unreadable"])
    print("written:", args.out)


if __name__ == "__main__":
    main()
