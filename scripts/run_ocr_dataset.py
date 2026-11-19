"""Run preprocessing -> CRAFT -> VietOCR on every MC-OCR receipt and write one OCR JSON per image (no LayoutXLM).

The output folder holds ``preprocessed_images/<id>.jpg`` and ``ocr_annotations/<id>.json``, the inputs of
``build_token_dataset.py`` and ``finetune_vietocr.py build``. A finished image is skipped when the command is run again
(restart after a disconnect).

``--weights`` reads with a fine-tuned VietOCR state dict; ``--nested-threshold`` overrides the detection setting.
Everything else comes from ``configs/model.yaml``, so a run without options reproduces the OCR of the token dataset v1.

Example (Kaggle, GPU):
    python scripts/run_ocr_dataset.py --images $IMG_RAW --split-manifest /kaggle/working/dataset/mcocr_manifest_split.csv \
        --out /kaggle/working/ocr_v1_regen
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.pipeline.runner import OutputDirs, ReceiptPipeline  # noqa: E402
from src.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402
from src.preprocessing.token_dataset import read_split_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with the ORIGINAL MC-OCR photos")
    ap.add_argument("--split-manifest", required=True, help="only the images listed here are processed")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="fine-tuned VietOCR state dict (default: pretrained)")
    ap.add_argument("--nested-threshold", type=float, default=None)
    ap.add_argument("--merge-lines", action="store_true",
                    help="read touching boxes of one line as one segment (src/detection/line_merge.py)")
    ap.add_argument("--merge-max-gap", type=float, default=None, help="largest merged gap, in text heights (1.0)")
    ap.add_argument("--config-dir", default=str(ROOT / "configs"))
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda but no GPU is visible (Kaggle: Accelerator = GPU T4). On CPU the whole dataset takes "
                 "hours; pass --device cpu to run it anyway.")
    if device == "cpu" and args.device == "auto":
        print("[ocr] WARNING: no GPU visible, running on CPU (several times slower)", flush=True)
    m = load_config(args.config_dir)["model"]
    det, rec = dict(m["detection"]), dict(m["recognition"])
    if args.nested_threshold is not None:
        det["nested_threshold"] = args.nested_threshold
    if args.merge_lines:
        det["merge_lines"] = True
    if args.merge_max_gap is not None:
        det["merge_max_gap"] = args.merge_max_gap
    if args.weights:
        rec["weights"] = args.weights

    from src.detection.craft_detector import CraftDetector
    from src.recognition.vietocr_recognizer import VietOCRRecognizer

    wanted = {i for i, s in read_split_manifest(args.split_manifest).items() if s in set(args.splits.split(","))}
    paths = sorted(os.path.join(args.images, f) for f in os.listdir(args.images) if f in wanted)
    missing = len(wanted) - len(paths)
    if args.limit:
        paths = paths[:args.limit]
    print(f"[ocr] {len(paths)} images (missing on disk: {missing}) | device={device} | weights={args.weights or 'pretrained'}"
          f" | nested_threshold={det.get('nested_threshold')} | merge_lines={bool(det.get('merge_lines'))}", flush=True)

    dirs = OutputDirs.from_base(args.out)
    pipeline = ReceiptPipeline(ImagePreprocessor(**m["preprocessing"]), CraftDetector(det, device=device),
                               VietOCRRecognizer(rec, device=device, batch_size=rec["batch_size"]), None, dirs)
    stats, failed = pipeline.run(paths)
    with open(os.path.join(args.out, "run_settings.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "detection": det, "recognition": rec, "stats": stats,
                   "failed": [a for a, _ in failed], "missing_on_disk": missing}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
