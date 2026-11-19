"""Digit benchmark for VietOCR weights on VALIDATION crops (the test split is never read).

Groups (each from a pair folder written by ``finetune_vietocr.py build`` / ``numbers``):

* ``total_region`` / ``total_line``: the annotated TOTAL_COST crops (exact MC-OCR labels, independent of any OCR run):
  the unbiased number benchmark and the gate;
* ``number``: item numbers of verified validation receipts (their OCR was already right: a regression check);
* ``synth``: synthetic numbers rendered with another seed than training (how well digits are read in general).

Metrics: CER, exact string, and ``digits`` (same digits in the same order, separators ignored: what decides an amount).

Example:
    python scripts/eval_numbers.py --pairs ocr_pairs --numbers number_pairs --synth synth_pairs \
        --weights round1=vietocr_ft/vietocr_finetuned.pth --weights digits=vietocr_num/vietocr_finetuned.pth \
        --out eval/numbers_val.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition.finetune import cer_summary  # noqa: E402
from src.recognition.finetune_data import load_pairs  # noqa: E402


def digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def number_summary(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    """``cer_summary`` plus the share of crops whose digits are all right."""
    out = dict(cer_summary(refs, hyps))
    out["digits"] = sum(digits(r) == digits(h) for r, h in zip(refs, hyps)) / max(1, len(refs))
    return out


def benchmark_groups(pairs: str, numbers: str, synth: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    if pairs:
        rows = load_pairs(os.path.join(pairs, "validation"))
        for kind in ("region", "line"):
            groups[f"total_{kind}"] = [r for r in rows if r["label"] == "TOTAL_COST" and r["kind"] == kind]
    if numbers:
        groups["number"] = [r for r in load_pairs(os.path.join(numbers, "validation")) if r["kind"] == "number"]
    if synth:
        groups["synth"] = load_pairs(os.path.join(synth, "validation"))
    return {k: v for k, v in groups.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="", help="folder of finetune_vietocr.py build (annotated TOTAL_COST crops)")
    ap.add_argument("--numbers", default="", help="folder of verified item numbers (finetune_vietocr.py numbers)")
    ap.add_argument("--synth", default="", help="folder of synthetic numbers (finetune_vietocr.py numbers)")
    ap.add_argument("--weights", action="append", required=True, help="name=state_dict, or name=pretrained")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from src.recognition.finetune import VietOCRFineTuner, load_vietocr
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    groups = benchmark_groups(args.pairs, args.numbers, args.synth)
    report: Dict[str, Dict[str, dict]] = {}
    for spec in args.weights:
        name, path = spec.split("=", 1)
        tuner = VietOCRFineTuner(load_vietocr(device, None if path == "pretrained" else path))
        report[name] = {}
        for g, rows in groups.items():
            report[name][g] = number_summary([r["text"] for r in rows], tuner.read(rows))
        del tuner
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"split": "validation", "groups": {g: len(r) for g, r in groups.items()}, "results": report}, f,
                  ensure_ascii=False, indent=2)
    print("| Weights | Group (n) | CER % | Exact % | Digits right % |\n|---|---|---|---|---|")
    for name, res in report.items():
        for g, s in res.items():
            print(f"| {name} | {g} ({len(groups[g])}) | {100 * s['cer']:.2f} | {100 * s['exact']:.1f} | {100 * s['digits']:.1f} |")
    print("written:", args.out)


if __name__ == "__main__":
    main()
