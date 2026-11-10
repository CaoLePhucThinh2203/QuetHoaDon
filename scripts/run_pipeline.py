"""Run the receipt pipeline on ONE image from the terminal and print / save the result as JSON.

Stages: preprocess -> CRAFT detection -> VietOCR recognition (pretrained weights are downloaded on first use).
LayoutXLM features and field extraction are not part of this command (see README, "Pipeline status").

Example:
    python scripts/run_pipeline.py --image assets/samples/sample_receipt.jpg --output outputs/sample_receipt.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.pipeline.runner import ReceiptPipeline  # noqa: E402
from src.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=str(ROOT / "assets/samples/sample_receipt.jpg"), help="Receipt image")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"), help="Folder with config/dataset/model.yaml")
    ap.add_argument("--output", default=None, help="Optional JSON file for the result")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args()


def build_pipeline(model_cfg: dict, device: str) -> ReceiptPipeline:
    from src.detection.craft_detector import CraftDetector
    from src.recognition.vietocr_recognizer import VietOCRRecognizer

    rec = model_cfg["recognition"]
    return ReceiptPipeline(
        preprocessor=ImagePreprocessor(**model_cfg["preprocessing"]),
        detector=CraftDetector(model_cfg["detection"], device=device),
        recognizer=VietOCRRecognizer(rec, device=device, batch_size=rec["batch_size"]),
        extractor=None,
    )


def main() -> None:
    args = parse_args()
    image = Path(args.image)
    if not image.is_file():
        sys.exit(f"Image not found: {image}")

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading models (device={device}, config={args.config_dir})", file=sys.stderr)
    pipeline = build_pipeline(load_config(args.config_dir)["model"], device)

    print(f"[*] Processing {image.name}", file=sys.stderr)
    result = pipeline.predict(image)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[+] Saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
