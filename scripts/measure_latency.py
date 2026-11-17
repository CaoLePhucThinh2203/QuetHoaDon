"""Latency and size of the extraction stage for B0, M1a and M1b on GPU and CPU (answers the latency part of RQ3).

For every receipt the timed block is the whole LayoutXLM stage: window building, image loading, processor, forward
pass, merging of the windows and decoding into the four fields (the same decoding that scored the results). The first
``--warmup`` receipts are run but not counted. OCR stages (preprocessing, CRAFT, VietOCR) are not re-run: with
``--ocr-dir`` the script averages the ``timing_sec`` already stored in the OCR JSON of the same receipts.

Examples:
    python scripts/measure_latency.py --token-dir $TOK --image-dir $IMG --ocr-dir pipeline_results/ocr_annotations \
        --checkpoint M1b=runs/m1b_v2_*/best --checkpoint M1a=runs/m1a_*/best --b0-model b0_cache/b0_k32.pkl \
        --out eval/latency.json
Scoring is not done here, so no labels are used and any split may be timed.
"""
import argparse
import glob
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, LayoutLMv2Model  # noqa: E402

from src.baseline.b0 import embed_records, predict_b0  # noqa: E402
from src.evaluation.latency import dir_size_mb, format_latency_table, mean_stage_seconds, summarize_times  # noqa: E402
from src.extraction.rules import fill_missing  # noqa: E402
from src.training.data import WindowDataset, build_windows, load_jsonl  # noqa: E402
from src.training.engine import DEFAULTS, load_model, predict_dataset  # noqa: E402
from src.training.field_eval import decode_predictions_soft  # noqa: E402


def time_receipts(run_one, records, warmup):
    from tqdm.auto import tqdm

    times = []
    for i, rec in enumerate(tqdm(records, desc="latency", leave=False)):
        t0 = time.perf_counter()
        run_one(rec)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return times


def decode(rec, merged):
    soft = decode_predictions_soft([rec], [merged])
    return fill_missing([rec], soft)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--checkpoint", action="append", default=[], help="NAME=path to a 'best' folder (repeatable)")
    ap.add_argument("--b0-model", default=None, help="b0_k*.pkl written by run_baseline_b0.py")
    ap.add_argument("--ocr-dir", default=None, help="OCR JSON folder, to report the mean time of the OCR stages")
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--n-gpu", type=int, default=60, help="Receipts timed on the GPU")
    ap.add_argument("--n-cpu", type=int, default=15, help="Receipts timed on the CPU (much slower)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_jsonl(str(Path(args.token_dir) / f"{args.split}.jsonl"))
    model_name = DEFAULTS["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    systems = []
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        matches = sorted(glob.glob(path))
        if not name or not matches:
            sys.exit(f"--checkpoint expects NAME=path and the path must exist: {item}")
        systems.append((name, matches[-1]))

    rows = []
    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        if device == "cuda" and not torch.cuda.is_available():
            print("[skip] no GPU available, cuda timings are not measured", file=sys.stderr)
            continue
        n = (args.n_gpu if device == "cuda" else args.n_cpu) + args.warmup
        recs = records[:n]

        for name, ckpt in systems:
            model = load_model(ckpt, False, device).eval()

            def run_one(rec, model=model, device=device):
                windows = build_windows([rec], processor.tokenizer, DEFAULTS["max_tokens"], DEFAULTS["overlap_tokens"])
                ds = WindowDataset([rec], windows, processor, args.image_dir, DEFAULTS["max_length"])
                merged, _ = predict_dataset(model, ds, [rec], device, use_amp=True, batch_size=8, num_workers=0)
                decode(rec, merged[0])

            times = time_receipts(run_one, recs, args.warmup)
            rows.append({"system": name, "device": device, **summarize_times(times),
                         "parameters": int(sum(p.numel() for p in model.parameters())),
                         "size_mb": dir_size_mb(ckpt)})
            print(f"[{name}/{device}] {rows[-1]['mean_ms']:.1f} ms mean over {rows[-1]['n']} receipts", file=sys.stderr)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        if args.b0_model:
            with open(args.b0_model, "rb") as f:
                b0 = pickle.load(f)
            encoder = LayoutLMv2Model.from_pretrained(model_name).to(device).eval()

            def run_b0(rec, encoder=encoder, device=device):
                emb = embed_records([rec], args.image_dir, model_name, DEFAULTS["max_tokens"], DEFAULTS["overlap_tokens"],
                                    batch_size=8, num_workers=0, encoder=encoder, processor=processor, device=device)
                merged, _, _ = predict_b0(b0, emb, [rec])
                decode(rec, merged[0])

            times = time_receipts(run_b0, recs, args.warmup)
            rows.append({"system": "B0", "device": device, **summarize_times(times),
                         "parameters": int(sum(p.numel() for p in encoder.parameters())),
                         "size_mb": dir_size_mb(args.b0_model)})
            print(f"[B0/{device}] {rows[-1]['mean_ms']:.1f} ms mean over {rows[-1]['n']} receipts", file=sys.stderr)
            del encoder
            if device == "cuda":
                torch.cuda.empty_cache()

    ocr_stage = None
    if args.ocr_dir:
        timings = []
        for rec in records[:args.n_gpu + args.warmup]:
            path = os.path.join(args.ocr_dir, f"{rec['file_id']}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    timings.append(json.load(f).get("timing_sec"))
        ocr_stage = {"receipts": len(timings), "mean_sec": mean_stage_seconds(timings),
                     "note": "stored when the OCR run was made; hardware of that run applies"}

    report = {
        "split": args.split, "warmup": args.warmup, "rows": rows, "ocr_stage_seconds": ocr_stage,
        "environment": {"torch": torch.__version__, "python": platform.python_version(), "cpu_count": os.cpu_count(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "what_is_timed": "window building + image loading + processor + forward + merge + decoding, per receipt",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(format_latency_table(rows))
    if ocr_stage:
        print("OCR stages (mean seconds per receipt):", {k: round(v, 3) for k, v in ocr_stage["mean_sec"].items()})
    print("written:", args.out)


if __name__ == "__main__":
    main()
