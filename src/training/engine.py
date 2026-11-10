"""Training and prediction loop for the LayoutXLM token classifier (needs torch + transformers; runs on Colab).

Design choices
* the loss is computed here (cross-entropy over the first sub-token of every word, ``-100`` ignored) so class
  weights and label smoothing can be tried;
* fp16 autocast + gradient scaling, gradient accumulation, AdamW with linear warm-up / decay;
* after every epoch the validation split is scored with entity-level F1 (``seqmetrics``); the best checkpoint is
  kept and training stops early when the F1 stops improving;
* a ``run.json`` with the configuration, versions, per-epoch history and the best epoch is rewritten after every
  epoch (task 0.5: experiment log).
The test split is never touched here.
"""
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.preprocessing.token_dataset import ID2LABEL, LABEL2ID, LABELS
from src.training.data import WindowDataset, build_windows
from src.training.seqmetrics import entity_scores
from src.training.windows import merge_window_probs

NUM_LABELS = len(LABELS)
_NOT_MODEL_INPUT = ("labels", "doc", "start")

DEFAULTS: Dict[str, Any] = {
    "model_name": "microsoft/layoutxlm-base",
    "epochs": 15,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "batch_size": 4,
    "grad_accum": 2,
    "grad_clip": 1.0,
    "max_length": 512,
    "max_tokens": 480,          # sub-token budget of a window (leaves room for <s> </s> and counting errors)
    "overlap_tokens": 100,
    "patience": 3,
    "seed": 42,
    "fp16": True,
    "num_workers": 2,
    "class_weight": "none",     # "none" | "sqrt_inverse"
    "label_smoothing": 0.0,
    "freeze_encoder": False,    # True = model M1a (train the classification head only)
}


# ---------------------------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(records: Sequence[Dict[str, Any]], mode: str) -> Optional[torch.Tensor]:
    """Loss weights per label from the training words. ``sqrt_inverse`` = (total / (C * count)) ** 0.5."""
    if mode == "none":
        return None
    if mode != "sqrt_inverse":
        raise ValueError(f"unknown class_weight mode: {mode}")
    counts = np.zeros(NUM_LABELS, dtype=np.float64)
    for rec in records:
        for tag in rec["labels"]:
            counts[LABEL2ID[tag]] += 1
    total = counts.sum()
    weights = np.ones(NUM_LABELS, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = np.sqrt(total / (NUM_LABELS * counts[nonzero]))
    return torch.tensor(weights, dtype=torch.float32)


def tags_from_probs(probs: np.ndarray) -> List[str]:
    return [ID2LABEL[int(i)] for i in probs.argmax(axis=1)]


def load_model(model_name: str, freeze_encoder: bool = False, device: str = "cuda"):
    """``LayoutLMv2ForTokenClassification`` (the class LayoutXLM checkpoints load into) with a fresh 9-way head."""
    from transformers import LayoutLMv2ForTokenClassification

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
    if freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return model.to(device)


def _model_inputs(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if k not in _NOT_MODEL_INPUT}


def _git_commit() -> Optional[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _write_json_atomic(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------- prediction
@torch.inference_mode()
def predict_dataset(model, dataset: WindowDataset, records: Sequence[Dict[str, Any]], device: str,
                    use_amp: bool = True, batch_size: int = 8, num_workers: int = 2
                    ) -> Tuple[List[Dict[str, Any]], float]:
    """Per-receipt word probabilities (windows merged) and the mean cross-entropy over all labelled words.

    Returns ``([{"probs": [n_words, 9], "covered": [n_words]}], mean_loss)``.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per_doc: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    loss_sum, loss_n = 0.0, 0
    for batch in loader:
        labels = batch["labels"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp and device == "cuda"):
            logits = model(**_model_inputs(batch, device)).logits
        logits = logits.float().cpu()
        loss_sum += F.cross_entropy(logits.reshape(-1, NUM_LABELS), labels.reshape(-1),
                                    ignore_index=-100, reduction="sum").item()
        loss_n += int((labels != -100).sum())
        probs = logits.softmax(dim=-1).numpy()
        for b in range(labels.shape[0]):
            pos = (labels[b] != -100).nonzero(as_tuple=True)[0].numpy()
            per_doc.setdefault(int(batch["doc"][b]), []).append((int(batch["start"][b]), probs[b][pos]))

    merged = []
    for d, rec in enumerate(records):
        probs, covered = merge_window_probs(len(rec["words"]), NUM_LABELS, per_doc.get(d, []))
        merged.append({"probs": probs, "covered": covered})
    return merged, loss_sum / max(loss_n, 1)


def evaluate_records(model, records: Sequence[Dict[str, Any]], processor: Any, image_dir: str, cfg: Dict[str, Any],
                     device: str) -> Dict[str, Any]:
    """Predict ``records`` and score them. Used for validation; the same call evaluates any split later."""
    windows = build_windows(records, processor.tokenizer, cfg["max_tokens"], cfg["overlap_tokens"])
    ds = WindowDataset(records, windows, processor, image_dir, cfg["max_length"])
    merged, loss = predict_dataset(model, ds, records, device, cfg["fp16"], cfg["batch_size"] * 2, cfg["num_workers"])
    pred = [tags_from_probs(m["probs"]) for m in merged]
    gold = [r["labels"] for r in records]
    scores = entity_scores(gold, pred)
    n_words = sum(len(r["words"]) for r in records)
    uncovered = sum(int((~m["covered"]).sum()) for m in merged)
    return {"loss": loss, "scores": scores, "predictions": merged, "uncovered_words": uncovered, "n_words": n_words}


# ---------------------------------------------------------------------------------- training
def train_model(cfg: Dict[str, Any], train_records: Sequence[Dict[str, Any]], val_records: Sequence[Dict[str, Any]],
                image_dir: str, out_dir: str) -> Dict[str, Any]:
    """Fine-tune LayoutXLM; returns the run log (also written to ``out_dir/run.json``)."""
    import transformers
    from transformers import AutoProcessor

    cfg = {**DEFAULTS, **cfg}
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg["fp16"]) and device == "cuda"
    t_start = time.time()

    processor = AutoProcessor.from_pretrained(cfg["model_name"], apply_ocr=False)
    train_windows = build_windows(train_records, processor.tokenizer, cfg["max_tokens"], cfg["overlap_tokens"])
    ds_train = WindowDataset(train_records, train_windows, processor, image_dir, cfg["max_length"])
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        generator=gen, pin_memory=(device == "cuda"))

    model = load_model(cfg["model_name"], cfg["freeze_encoder"], device)
    weights = class_weights(train_records, cfg["class_weight"])
    if weights is not None:
        weights = weights.to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if ("bias" in name or "LayerNorm" in name or "layer_norm" in name) else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(loader) / cfg["grad_accum"])
    total_steps = max(1, steps_per_epoch * cfg["epochs"])
    warmup = int(cfg["warmup_ratio"] * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    windows_per_receipt = Counter(w[0] for w in train_windows)
    n_multi = sum(1 for c in windows_per_receipt.values() if c > 1)
    run: Dict[str, Any] = {
        "run_id": os.path.basename(os.path.normpath(out_dir)),
        "config": cfg,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "git_commit": _git_commit()},
        "device": device,
        "data": {"train_receipts": len(train_records), "val_receipts": len(val_records),
                 "train_windows": len(train_windows), "train_receipts_needing_several_windows": n_multi,
                 "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "history": [], "best_epoch": None, "best_val_f1": -1.0, "status": "running",
    }
    run_path = os.path.join(out_dir, "run.json")
    _write_json_atomic(run, run_path)
    print(f"[run] {run['run_id']} | device={device} | train windows={len(train_windows)} | "
          f"trainable params={run['data']['trainable_parameters']:,}")

    bad_epochs = 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**_model_inputs(batch, device)).logits
            loss = F.cross_entropy(logits.float().reshape(-1, NUM_LABELS), labels.reshape(-1), weight=weights,
                                   ignore_index=-100, label_smoothing=cfg["label_smoothing"])
            scaler.scale(loss / cfg["grad_accum"]).backward()
            loss_sum += float(loss.detach())
            n_batches += 1
            if (i + 1) % cfg["grad_accum"] == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        ev = evaluate_records(model, val_records, processor, image_dir, cfg, device)
        micro = ev["scores"]["micro"]
        entry = {
            "epoch": epoch, "train_loss": loss_sum / max(1, n_batches), "val_loss": ev["loss"],
            "val_precision": micro["precision"], "val_recall": micro["recall"], "val_f1": micro["f1"],
            "val_f1_by_field": {f: ev["scores"][f]["f1"] for f in ev["scores"] if f != "micro"},
            "val_uncovered_words": ev["uncovered_words"], "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        run["history"].append(entry)
        improved = entry["val_f1"] > run["best_val_f1"]
        if improved:
            run["best_val_f1"], run["best_epoch"] = entry["val_f1"], epoch
            model.save_pretrained(os.path.join(out_dir, "best"))
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"[epoch {epoch:>2}] train_loss={entry['train_loss']:.4f} val_loss={entry['val_loss']:.4f} "
              f"val_F1={entry['val_f1']:.4f} (P={entry['val_precision']:.3f} R={entry['val_recall']:.3f}) "
              f"{entry['seconds']}s{'  *best*' if improved else ''}")
        _write_json_atomic(run, run_path)
        if bad_epochs >= cfg["patience"]:
            run["status"] = "early_stopped"
            print(f"[run] early stop: no F1 improvement for {cfg['patience']} epochs")
            break
    else:
        run["status"] = "finished"

    run["total_seconds"] = round(time.time() - t_start, 1)
    _write_json_atomic(run, run_path)
    return run
