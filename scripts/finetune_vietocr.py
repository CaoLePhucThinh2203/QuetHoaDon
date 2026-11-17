"""Fine-tune VietOCR on MC-OCR receipt crops and measure it against the pretrained model (validation only).

Two steps:

``build``  make the (crop, text) pairs of the train and validation splits from an OCR run of the pipeline
           (``scripts/run_ocr_dataset.py``: processed images + OCR JSON) and the MC-OCR annotation CSV.
``train``  fine-tune, keep the epoch with the lowest validation CER on the CRAFT-region pairs, then read every
           validation pair with the pretrained and the fine-tuned model and write the comparison (CER per field and
           per pair kind, paired bootstrap over receipts; negative difference = fine-tuned is better).

The test split is never written or read here.

Example (Kaggle, GPU):
    python scripts/finetune_vietocr.py build --csv $CSV --ocr-dir $OCR/ocr_annotations --image-dir $OCR/preprocessed_images \
        --split-manifest /kaggle/working/dataset/mcocr_manifest_split.csv --out /kaggle/working/ocr_pairs
    python scripts/finetune_vietocr.py train --pairs /kaggle/working/ocr_pairs --out /kaggle/working/vietocr_ft
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ocr_ablation import paired_cer_difference  # noqa: E402
from src.recognition.finetune_data import RegionPairConfig, build_pair_set, load_pairs  # noqa: E402


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_build(args: argparse.Namespace) -> None:
    from vietocr.tool.config import Cfg
    vocab = Cfg.load_config_from_name("vgg_transformer")["vocab"]
    cfg = RegionPairConfig(args.min_containment, args.max_line_cer, args.max_segment_cer)
    report = build_pair_set(args.csv, args.ocr_dir, args.image_dir, args.split_manifest, ["train", "validation"],
                            args.out, vocab, cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _strip(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k != "hyps"}


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from src.recognition.finetune import VietOCRFineTuner, load_vietocr, per_receipt_edits

    torch.manual_seed(args.seed)
    device = _device(args.device)
    train_rows = load_pairs(os.path.join(args.pairs, "train"))
    val_rows = load_pairs(os.path.join(args.pairs, "validation"))
    if args.kinds != "line,region":
        keep = set(args.kinds.split(","))
        train_rows = [r for r in train_rows if r["kind"] in keep]
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    os.makedirs(args.out, exist_ok=True)
    weights = os.path.join(args.out, "vietocr_finetuned.pth")
    print(f"[ft] device={device} train pairs={len(train_rows)} validation pairs={len(val_rows)}", flush=True)

    base = VietOCRFineTuner(load_vietocr(device, args.init_weights), args.batch_size, args.seed)
    before = base.evaluate(val_rows)
    del base
    tuner = VietOCRFineTuner(load_vietocr(device, args.init_weights), args.batch_size, args.seed)
    fit = tuner.fit(train_rows, val_rows, weights, args.epochs, args.lr, args.patience,
                    log=lambda m: print(m, flush=True))
    after = tuner.evaluate(val_rows)

    paired = {"all": paired_cer_difference(per_receipt_edits(val_rows, before["hyps"]),
                                           per_receipt_edits(val_rows, after["hyps"]), args.n_boot)}
    for kind in sorted({r["kind"] for r in val_rows}):
        idx = [i for i, r in enumerate(val_rows) if r["kind"] == kind]
        rows = [val_rows[i] for i in idx]
        paired[kind] = paired_cer_difference(per_receipt_edits(rows, [before["hyps"][i] for i in idx]),
                                             per_receipt_edits(rows, [after["hyps"][i] for i in idx]), args.n_boot)
    examples = [{"img_id": r["img_id"], "label": r["label"], "kind": r["kind"], "gold": r["text"],
                 "pretrained": b, "finetuned": a}
                for r, b, a in zip(val_rows, before["hyps"], after["hyps"]) if b != a][:200]
    report = {"split": "validation", "init_weights": args.init_weights, "train_pairs": len(train_rows), "validation_pairs": len(val_rows),
              "settings": {k: getattr(args, k) for k in ("epochs", "lr", "batch_size", "patience", "seed", "kinds")},
              "fit": fit, "pretrained": _strip(before), "finetuned": _strip(after),
              "paired_finetuned_minus_pretrained": paired, "changed_examples": examples}
    with open(os.path.join(args.out, "finetune_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n| Validation pairs | Pretrained CER % | Fine-tuned CER % | Pretrained exact % | Fine-tuned exact % |")
    print("|---|---|---|---|---|")
    rows = [("all", before["all"], after["all"])]
    rows += [(f"kind {k}", before["by_kind"][k], after["by_kind"][k]) for k in before["by_kind"]]
    rows += [(f"field {k}", before["by_label"][k], after["by_label"][k]) for k in before["by_label"]]
    for name, b, a in rows:
        print(f"| {name} ({b['n']}) | {100 * b['cer']:.2f} | {100 * a['cer']:.2f} | {100 * b['exact']:.1f} | "
              f"{100 * a['exact']:.1f} |")
    print("\npaired CER difference (fine-tuned - pretrained, percentage points, 95 % bootstrap over receipts):")
    for k, d in paired.items():
        print(f"  {k}: {100 * d['diff']:+.2f} [{100 * d['low']:+.2f}, {100 * d['high']:+.2f}] over {d['n']} receipts")
    print(f"best epoch {fit['best_epoch']} | weights: {weights}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write the crop / text pairs of train and validation")
    b.add_argument("--csv", required=True)
    b.add_argument("--ocr-dir", required=True, help="OCR JSON folder of the pipeline run")
    b.add_argument("--image-dir", required=True, help="processed images of the same run (<file_id>.jpg)")
    b.add_argument("--split-manifest", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--min-containment", type=float, default=RegionPairConfig.min_containment)
    b.add_argument("--max-line-cer", type=float, default=RegionPairConfig.max_line_cer)
    b.add_argument("--max-segment-cer", type=float, default=RegionPairConfig.max_segment_cer)

    t = sub.add_parser("train", help="fine-tune and compare with the pretrained model on validation")
    t.add_argument("--pairs", required=True, help="folder written by 'build'")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--kinds", default="line,region", help="pair kinds used for training: line,region | region | line")
    t.add_argument("--max-train", type=int, default=None, help="use only the first N training pairs (smoke test)")
    t.add_argument("--n-boot", type=int, default=1000)
    t.add_argument("--init-weights", default=None,
                   help="start from this fine-tuned state dict (round 2); the 'pretrained' columns of the report are then "
                        "this starting model, so the gate measures the gain of this round only")
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = ap.parse_args()
    cmd_build(args) if args.cmd == "build" else cmd_train(args)


if __name__ == "__main__":
    main()
