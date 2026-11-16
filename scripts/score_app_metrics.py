"""Strict "reading accuracy" (the view of an app user) for saved evaluation reports (CPU, no model).

Reads the final predictions stored in each report (``per_receipt_soft_rules``) and the annotations of the token dataset
(``gt``), and prints exact-match accuracy with 95 % intervals per field (see ``src/evaluation/app_metrics.py``).

Example:
    python scripts/score_app_metrics.py --token-dir $TOK --split validation \
        --report M1b=eval/m1b_v2_val.json --report B0=eval/b0_val.json --out eval/app_metrics_val.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.app_metrics import app_scores, format_app_table  # noqa: E402
from src.evaluation.reproduction import reference_predictions  # noqa: E402
from src.training.data import load_jsonl  # noqa: E402
from src.training.field_eval import gold_fields  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--allow-test", action="store_true", help="Required for the test split (it was scored once in Week 4)")
    ap.add_argument("--report", action="append", required=True, help="NAME=path to an evaluation report (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        sys.exit("The test split is only read for the final evaluation. Re-run with --allow-test.")

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    gts = gold_fields(records)
    result, tables = {}, []
    for item in args.report:
        name, _, path = item.partition("=")
        if not name or not path:
            sys.exit(f"--report expects NAME=path, got {item}")
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if report.get("split") not in (None, args.split):
            sys.exit(f"{path} is a {report.get('split')} report, not {args.split}")
        scores = app_scores(gts, reference_predictions(report))
        result[name] = scores
        tables.append(format_app_table(scores, f"{name} ({args.split}, {len(records)} receipts)"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    text = "\n".join(tables)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    print(text)
    print("written:", args.out)


if __name__ == "__main__":
    main()
