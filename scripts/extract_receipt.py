"""Receipt photo -> SELLER, ADDRESS, TIMESTAMP, TOTAL_COST with the fine-tuned LayoutXLM (end-to-end demo, GPU).

Runs preprocessing, CRAFT, VietOCR and the token classifier, then prints one JSON document. Fields are decoded with the
same functions that produced the scored results (soft decoding + keyword / date rules).

Examples:
    # final configuration (fine-tuned VietOCR + M1b v3 + rules + items, short output for an app):
    python scripts/extract_receipt.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
        --nested-threshold 0.5 --postprocess --items --compact --image path/to/receipt.jpg
    python scripts/extract_receipt.py --checkpoint runs/m1b_v2_*/best --image path/to/receipt.jpg --output out.json
    # reuse an OCR result and its preprocessed image (no CRAFT / VietOCR needed):
    python scripts/extract_receipt.py --checkpoint runs/m1b_v2_*/best \
        --ocr-json pipeline_results/ocr_annotations/<id>.json --preprocessed-image pipeline_results/preprocessed_images/<id>.jpg
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.extractor import ReceiptExtractor, compact_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint folder (the 'best' folder); globs allowed")
    ap.add_argument("--image", default=None, help="Receipt photo (runs the OCR pipeline)")
    ap.add_argument("--ocr-json", default=None, help="Existing OCR result <id>.json (instead of --image)")
    ap.add_argument("--preprocessed-image", default=None, help="Preprocessed image <id>.jpg that belongs to --ocr-json")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--output", default=None, help="Optional JSON file for the result")
    ap.add_argument("--ocr-weights", default=None,
                    help="fine-tuned VietOCR state dict (use it with the M1b checkpoint trained on that OCR)")
    ap.add_argument("--postprocess", action="store_true",
                    help="add the time next to the date and clean SELLER / ADDRESS (Week-5 decoding)")
    ap.add_argument("--nested-threshold", type=float, default=None,
                    help="nested-box filter of CRAFT (the final configuration uses 0.5; configs/model.yaml has 0.8)")
    ap.add_argument("--items", action="store_true",
                    help="also read the item lines (name, quantity, unit price, amount) with the rule based reader")
    ap.add_argument("--compact", action="store_true",
                    help="short output for an app: only fields that were read, time split from the date, items if any")
    ap.add_argument("--show-words", action="store_true", help="Also print every word with its box and predicted tag")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_json = args.ocr_json is not None
    if use_json == (args.image is not None):
        sys.exit("Give either --image, or --ocr-json together with --preprocessed-image.")
    if use_json and not args.preprocessed_image:
        sys.exit("--ocr-json needs --preprocessed-image (the image the OCR boxes refer to).")

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        sys.exit(f"No checkpoint matches {args.checkpoint}")
    checkpoint = matches[-1]

    print(f"[*] Loading models from {checkpoint} (device={args.device})", file=sys.stderr)
    extractor = ReceiptExtractor.from_checkpoint(checkpoint, args.config_dir, args.device, with_ocr=not use_json,
                                                 postprocess=args.postprocess, ocr_weights=args.ocr_weights,
                                                 items=args.items, nested_threshold=args.nested_threshold)
    if use_json:
        with open(args.ocr_json, "r", encoding="utf-8") as f:
            result = extractor.extract_from_ocr(json.load(f), args.preprocessed_image)
    else:
        result = extractor.extract(args.image)

    if args.compact:
        result = compact_result(result)
    elif not args.show_words:
        result.pop("words", None)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[+] Saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
