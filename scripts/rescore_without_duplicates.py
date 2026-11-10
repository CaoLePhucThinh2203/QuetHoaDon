"""Re-score saved evaluation reports without the receipts that have a near-duplicate in another split (CPU).

Reads the ``per_receipt_soft_rules`` rows stored in each report, drops the receipts listed in ``leaking`` of the
duplicate report, and prints macro F1 with a bootstrap interval for every system on the remaining receipts.

Example:
    python scripts/rescore_without_duplicates.py --dups dup_report.json --report B0=eval/b0_test.json \
        --report M1a=eval/m1a_test.json --report M1b=eval/m1b_v2_test.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

FIELDS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")


def metrics(rows, idx):
    res = {}
    for f in FIELDS:
        tp = fp = fn = 0
        for i in idx:
            t, p, n = rows[i]["outcome"][f]
            tp, fp, fn = tp + t, fp + p, fn + n
        res[f] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    res["macro_f1"] = float(np.mean([res[f] for f in FIELDS]))
    n = sum(rows[i]["has_any"] for i in idx)
    res["receipt_rate"] = sum(rows[i]["all_ok"] for i in idx) / n if n else 0.0
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dups", required=True)
    ap.add_argument("--report", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    leaking = set(json.load(open(args.dups, encoding="utf-8"))["leaking"])
    rng = np.random.default_rng(0)
    out = {}
    print(f"{'system':<6} {'kept/all':>9}  macro F1 (all) -> (without duplicates) [95% CI]   receipt rate (all) -> (kept)")
    for item in args.report:
        name, path = item.split("=", 1)
        rep = json.load(open(path, encoding="utf-8"))
        rows = rep["per_receipt_soft_rules"]
        keep = [i for i, r in enumerate(rows) if r["file_id"] not in leaking]
        full, kept = metrics(rows, range(len(rows))), metrics(rows, keep)
        boots = np.array([metrics(rows, [keep[j] for j in rng.integers(0, len(keep), len(keep))])["macro_f1"]
                          for _ in range(args.n_boot)]) if keep else np.array([0.0])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out[name] = {"kept": len(keep), "all": len(rows), "full": full, "without_duplicates": kept, "macro_ci": [float(lo), float(hi)]}
        print(f"{name:<6} {len(keep):>4}/{len(rows):<4} {full['macro_f1']:.3f} -> {kept['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]"
              f"      {full['receipt_rate']:.3f} -> {kept['receipt_rate']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
