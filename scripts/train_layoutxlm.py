"""Fine-tune LayoutXLM for BIO token classification on the token dataset (run on a Colab GPU).

Only ``train.jsonl`` (fitting) and ``validation.jsonl`` (early stopping, model selection) are read. The test split
is never touched here: it is scored once, at the very end, by the evaluation script.

Example (M1b, the improved model):
    python scripts/train_layoutxlm.py \
        --token-dir /content/drive/MyDrive/quethoadon/token_dataset_v1 \
        --image-dir /content/drive/MyDrive/quethoadon/ocr_v1/preprocessed_images \
        --out /content/drive/MyDrive/quethoadon/runs --name m1b

M1a (frozen encoder, train the head only): add ``--freeze-encoder`` and ``--name m1a``.
Quick smoke test: ``--max-train-docs 16 --max-val-docs 8 --epochs 1``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data import load_jsonl  # noqa: E402
from src.training.engine import DEFAULTS, train_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True, help="Folder with train.jsonl and validation.jsonl")
    ap.add_argument("--image-dir", required=True, help="Folder with the preprocessed receipt images (<file_id>.jpg)")
    ap.add_argument("--out", required=True, help="Folder that receives one sub-folder per run (checkpoint + run.json)")
    ap.add_argument("--name", default="m1b", help="Run name prefix")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--class-weight", choices=["none", "sqrt_inverse"], default=DEFAULTS["class_weight"])
    ap.add_argument("--label-smoothing", type=float, default=DEFAULTS["label_smoothing"])
    ap.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"], help="Sub-token budget of a window")
    ap.add_argument("--overlap-tokens", type=int, default=DEFAULTS["overlap_tokens"])
    ap.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--freeze-encoder", action="store_true", help="Train only the classification head (M1a)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--max-train-docs", type=int, default=None, help="Use only the first N training receipts")
    ap.add_argument("--max-val-docs", type=int, default=None, help="Use only the first N validation receipts")
    args = ap.parse_args()

    token_dir = Path(args.token_dir)
    train = load_jsonl(str(token_dir / "train.jsonl"))
    val = load_jsonl(str(token_dir / "validation.jsonl"))
    if args.max_train_docs:
        train = train[:args.max_train_docs]
    if args.max_val_docs:
        val = val[:args.max_val_docs]

    cfg = {
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio, "patience": args.patience,
        "seed": args.seed, "class_weight": args.class_weight, "label_smoothing": args.label_smoothing,
        "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens, "num_workers": args.num_workers,
        "freeze_encoder": args.freeze_encoder, "fp16": not args.no_fp16,
    }
    out_dir = Path(args.out) / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = train_model(cfg, train, val, args.image_dir, str(out_dir))

    print(json.dumps({k: run[k] for k in ("run_id", "status", "best_epoch", "best_val_f1", "total_seconds")},
                     indent=2, ensure_ascii=False))
    print("checkpoint:", out_dir / "best", "| log:", out_dir / "run.json")


if __name__ == "__main__":
    main()
