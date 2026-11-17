"""RQ1: effect of preprocessing (and of synthetic degradations) on VietOCR text recognition, validation split only.

Each annotated region of the validation receipts is cropped from the original photo under four preprocessing
configurations (none, deskew, contrast, full) and read by VietOCR; CER and WER are computed against the annotated text,
overall and by image-quality third, with a paired bootstrap against "none". The test split is not offered on purpose.

Needs the original MC-OCR photos (not the already preprocessed ones), the annotation CSV and the split manifest.
Example (Kaggle, GPU):
    python scripts/ocr_preprocessing_ablation.py --csv <dataset>/mcocr_train_df.csv \
        --image-dir <dataset>/train_images/train_images --split-manifest outputs/dataset/mcocr_manifest_split.csv \
        --degradations none,blur,dark,noise --out eval/ocr_ablation.json --md eval/ocr_ablation.md
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.evaluation.ocr_ablation import (CONFIGS, DEGRADATIONS, build_report, format_markdown, load_items,  # noqa: E402
                                         run_ablation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="MC-OCR annotation CSV (mcocr_train_df.csv)")
    ap.add_argument("--image-dir", required=True, help="Folder with the ORIGINAL receipt photos")
    ap.add_argument("--split-manifest", required=True, help="mcocr_manifest_split.csv from prepare_dataset.py")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--degradations", default="none", help=f"comma separated, from {','.join(DEGRADATIONS)}")
    ap.add_argument("--max-receipts", type=int, default=None, help="Use only the first N receipts (smoke test)")
    ap.add_argument("--weights", default=None, help="fine-tuned VietOCR state dict (default: pretrained)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True, help="JSON report")
    ap.add_argument("--md", default=None, help="Markdown tables")
    args = ap.parse_args()

    degradations = [d.strip() for d in args.degradations.split(",") if d.strip()]
    unknown = [d for d in degradations if d not in DEGRADATIONS]
    if unknown:
        sys.exit(f"unknown degradation(s): {unknown}")

    from src.recognition.vietocr_recognizer import VietOCRRecognizer
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = load_config(args.config_dir)["model"]
    rec_cfg = dict(model_cfg["recognition"])
    if args.weights:
        rec_cfg["weights"] = args.weights
    recognizer = VietOCRRecognizer(rec_cfg, device=device, batch_size=rec_cfg["batch_size"])

    items = load_items(args.csv, args.split_manifest, "validation", args.image_dir, args.max_receipts)
    print(f"[*] {len(items)} validation receipts, degradations={degradations}, device={device}", file=sys.stderr)
    result = run_ablation(items, recognizer, model_cfg["preprocessing"]["target_max_dim"], CONFIGS, degradations)
    report = build_report(result, degradations, list(CONFIGS), args.n_boot)
    report.update({"split": "validation", "weights": args.weights, "degradations": degradations, "configs": CONFIGS, "device": device,
                   "recognizer_errors": getattr(recognizer, "n_errors", 0)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    table = format_markdown(report)
    if args.md:
        Path(args.md).write_text(table, encoding="utf-8")
    print(table)
    print("skipped:", report["skipped"], "| written:", args.out)


if __name__ == "__main__":
    main()
