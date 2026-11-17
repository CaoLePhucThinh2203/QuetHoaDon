"""Paired comparison of two decoded runs on the same receipts, strict reading accuracy (CPU, seconds).

Takes the ``decoded_*.json`` files written by ``decode_from_probs.py`` and the token dataset (for the annotations), scores
both with the same strict scorer and reports, per field and for "all fields right":

* the two counts and the paired difference in accuracy with a 95 % bootstrap interval over receipts;
* the exact McNemar test on the receipts where exactly one run is right (two-sided, binomial with p = 0.5).

Example:
    python scripts/compare_strict.py --token-dir phase1_eval/tok_ft_n05 --split test --allow-test \
        --a week4=phase1_eval/eval/decoded_v2_test.json --b final=phase1_eval/eval/decoded_v3_test.json \
        --out phase1_eval/eval/paired_test.json
"""
import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.app_metrics import app_scores  # noqa: E402
from src.evaluation.reproduction import reference_predictions  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact p-value for ``b_only`` vs ``a_only`` discordant receipts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_diff(pairs, n_boot=2000, seed=0):
    """95 % interval of mean(b) - mean(a) over (a, b) pairs of 0/1 outcomes."""
    rnd = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rnd.randrange(n)] for _ in range(n)]
        diffs.append(sum(b - a for a, b in sample) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--a", required=True, help="NAME=decoded json of the reference run")
    ap.add_argument("--b", required=True, help="NAME=decoded json of the run compared against it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is read only after the configuration is frozen. Re-run with --allow-test.")

    gts = {r["img_id"]: r["gt"] for r in load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))}
    runs = {}
    for spec in (args.a, args.b):
        name, _, path = spec.partition("=")
        with open(path, "r", encoding="utf-8") as f:
            runs[name] = app_scores(gts, reference_predictions(json.load(f)))
    (name_a, sa), (name_b, sb) = list(runs.items())

    receipts = sorted(set(sa["per_receipt"]) & set(sb["per_receipt"]))
    rows = {}
    for field in ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "ALL"):
        pairs = []
        for img in receipts:
            fa, fb = sa["per_receipt"][img], sb["per_receipt"][img]
            if field == "ALL":
                if fa and fb:
                    pairs.append((int(all(fa.values())), int(all(fb.values()))))
            elif field in fa and field in fb:
                pairs.append((int(fa[field]), int(fb[field])))
        a_only = sum(1 for a, b in pairs if a and not b)
        b_only = sum(1 for a, b in pairs if b and not a)
        low, high = bootstrap_diff(pairs) if pairs else (0.0, 0.0)
        rows[field] = {"n": len(pairs), f"{name_a}_correct": sum(a for a, _ in pairs),
                       f"{name_b}_correct": sum(b for _, b in pairs), f"only_{name_a}": a_only, f"only_{name_b}": b_only,
                       "diff": (sum(b - a for a, b in pairs) / len(pairs)) if pairs else 0.0, "ci_low": low,
                       "ci_high": high, "mcnemar_p": mcnemar_exact(b_only, a_only)}
    report = {"split": args.split, "a": name_a, "b": name_b, "fields": rows,
              "alone": {name_a: {k: v["correct"] for k, v in sa["fields"].items()},
                        name_b: {k: v["correct"] for k, v in sb["fields"].items()}}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"### Paired strict reading accuracy on {args.split}: {name_b} minus {name_a}")
    print(f"| Field | n | {name_a} right | {name_b} right | only {name_a} | only {name_b} | Difference | 95 % interval | McNemar p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for field, r in rows.items():
        print(f"| {field} | {r['n']} | {r[f'{name_a}_correct']} | {r[f'{name_b}_correct']} | {r[f'only_{name_a}']} | "
              f"{r[f'only_{name_b}']} | {100 * r['diff']:+.1f} pts | [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | "
              f"{r['mcnemar_p']:.4f} |")
    print("written:", args.out)


if __name__ == "__main__":
    main()
