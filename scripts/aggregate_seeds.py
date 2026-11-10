"""Mean and standard deviation over several training seeds (CPU, reads evaluation reports).

Every report comes from ``evaluate_layoutxlm.py`` on the same split, one per seed. The point values stored under
``bootstrap_soft_rules`` (final decoding) are averaged; the standard deviation measures how much the result moves
between runs, which the bootstrap interval of one run does not.

Example:
    python scripts/aggregate_seeds.py --system M1b=eval/m1b_s42_test.json,eval/m1b_s1_test.json,eval/m1b_s2_test.json \
        --system M1a=eval/m1a_s42_test.json,eval/m1a_s1_test.json,eval/m1a_s2_test.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

METRICS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST", "macro_f1", "receipt_rate")


def summarize(paths):
    vals = {m: [] for m in METRICS}
    for p in paths:
        boot = json.load(open(p, encoding="utf-8"))["bootstrap_soft_rules"]
        for m in METRICS:
            vals[m].append(boot[m]["value"])
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "runs": len(v)}
            for m, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="NAME=report1.json,report2.json,... (repeatable)")
    ap.add_argument("--out", default=None, help="Markdown file to write")
    args = ap.parse_args()
    head = "| System | Runs | " + " | ".join(METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 2)]
    for item in args.system:
        name, paths = item.split("=", 1)
        s = summarize(paths.split(","))
        lines.append(f"| {name} | {s['macro_f1']['runs']} | " + " | ".join(f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f}" for m in METRICS) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
