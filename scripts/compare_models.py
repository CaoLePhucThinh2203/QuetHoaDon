"""Comparison table of several evaluated systems (CPU only, reads the JSON reports).

Each report comes from ``evaluate_layoutxlm.py`` or ``run_baseline_b0.py`` on the SAME split. The table uses the
final decoding (lenient matching, soft decoding, rules fill empty TOTAL_COST / TIMESTAMP) with 95 % bootstrap
intervals, and a paired bootstrap of every system against the baseline (same receipts resampled together).

Example:
    python scripts/compare_models.py --baseline B0 \
        --report B0=eval/b0_val.json --report M1a=eval/m1a_val.json --report M1b=eval/m1b_v2_val.json \
        --out eval/comparison_val.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import FIELDS  # noqa: E402
from src.training.field_eval import outcomes_from_rows, paired_bootstrap  # noqa: E402


def ci(v):
    return f"{v['value']:.3f} [{v['low']:.3f}, {v['high']:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", required=True, help="NAME=path/to/report.json (repeatable)")
    ap.add_argument("--baseline", required=True, help="NAME of the baseline report")
    ap.add_argument("--variant", choices=["soft", "soft_rules"], default="soft_rules",
                    help="soft = model only; soft_rules = model + rules fill empty fields (final system)")
    ap.add_argument("--out", default=None, help="Markdown file to write")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    reports = {}
    for item in args.report:
        name, path = item.split("=", 1)
        with open(path, encoding="utf-8") as f:
            reports[name] = json.load(f)
    v = args.variant
    L = [f"## Comparison ({v}, split = {next(iter(reports.values())).get('split')})", ""]
    head = "| System | " + " | ".join(f"{f} F1" for f in FIELDS) + " | Macro F1 | Receipt all-correct | Total exact / with gold | Token AUC | ms / receipt |"
    L += [head, "|" + "---|" * (head.count("|") - 1)]
    for name, r in reports.items():
        b = r[f"bootstrap_{v}"]
        amt = r[f"total_amount_{v}"]
        auc = r.get("token_auc", {}).get("macro")
        ms = r.get("seconds_per_receipt")
        cells = [ci(b[f]) for f in FIELDS] + [ci(b["macro_f1"]), ci(b["receipt_rate"]),
                                              f"{amt['exact']} / {amt['receipts_with_gt_total']}",
                                              f"{auc:.3f}" if auc is not None else "-",
                                              f"{ms * 1000:.0f}" if ms else "-"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    if "fields_oracle" in reports[args.baseline]:
        o = reports[args.baseline]["fields_oracle"]
        L += ["", "Ceiling with this OCR (gold tags decoded): " + ", ".join(f"{f} {o[f]['f1']:.3f}" for f in FIELDS)]

    base = outcomes_from_rows(reports[args.baseline][f"per_receipt_{v}"])
    L += ["", f"### Paired bootstrap against {args.baseline} (difference, 95 % interval, share of resamples better)", "",
          "| System | " + " | ".join(FIELDS) + " | Macro F1 | Receipt all-correct |", "|" + "---|" * 7]
    for name, r in reports.items():
        if name == args.baseline:
            continue
        pb = paired_bootstrap(base, outcomes_from_rows(r[f"per_receipt_{v}"]), args.n_boot)["metrics"]
        L.append(f"| {name} | " + " | ".join(
            f"{pb[k]['diff']:+.3f} [{pb[k]['low']:+.3f}, {pb[k]['high']:+.3f}] ({pb[k]['p_b_better']:.2f})"
            for k in (*FIELDS, "macro_f1", "receipt_rate")) + " |")

    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("written:", args.out)


if __name__ == "__main__":
    main()
