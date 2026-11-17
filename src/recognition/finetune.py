"""Fine-tune the pretrained VietOCR (vgg_transformer) on receipt line / region crops.

Why our own loop instead of ``vietocr.model.trainer.Trainer``: the Trainer imports ``imgaug`` (broken with NumPy 2, the
version on Kaggle), stores the data in LMDB and silently drops every width bucket with fewer samples than one batch.
This loop reuses VietOCR's own pieces (``Predictor`` to load the exact pretrained model and vocabulary the pipeline
uses, ``process_image`` for the input, ``LabelSmoothingLoss``), so a fine-tuned state dict drops straight into
``VietOCRRecognizer`` through the ``weights`` option.

Pure helpers (bucketing, target tensors, augmentation, CER summary) need only NumPy / OpenCV and are unit-tested;
everything that needs torch or vietocr is imported inside the functions.
"""
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import edit_distance, normalize_text
from src.recognition.finetune_data import INFER_PAD, padded_view

PAD_X_RANGE = (1, 6)         # training padding is drawn around the inference padding INFER_PAD = (3, 2)
PAD_Y_RANGE = (0, 4)


# ---------------------------------------------------------------------------------- pure helpers
def resized_width(w: int, h: int, height: int = 32, min_width: int = 32, max_width: int = 512) -> int:
    """Width VietOCR resizes a ``w`` x ``h`` crop to (same rule as ``vietocr.tool.translate.resize``)."""
    new_w = int(height * float(w) / float(max(h, 1)))
    new_w = int(math.ceil(new_w / 10) * 10)
    return min(max(new_w, min_width), max_width)


def width_buckets(widths: Sequence[int], batch_size: int, rng: Optional[np.random.Generator] = None
                  ) -> List[List[int]]:
    """Batches of indices whose images share one resized width (no padding needed); partial batches are kept.

    With ``rng`` the order inside buckets and the order of the batches are shuffled.
    """
    groups: Dict[int, List[int]] = {}
    for i, w in enumerate(widths):
        groups.setdefault(int(w), []).append(i)
    batches: List[List[int]] = []
    for w in sorted(groups):
        idx = list(groups[w])
        if rng is not None:
            rng.shuffle(idx)
        batches.extend(idx[k:k + batch_size] for k in range(0, len(idx), batch_size))
    if rng is not None:
        order = rng.permutation(len(batches))
        batches = [batches[k] for k in order]
    return batches


def target_tensors(encoded: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tgt_input (T, N), tgt_output (N, T), padding_mask (N, T))`` exactly as VietOCR's ``Collator`` builds them.

    Every sequence is ``[sos] + chars + [eos]``; padding id is 0. The mask is True on positions the decoder ignores.
    """
    max_len = max(len(e) for e in encoded)
    tgt, weights = [], []
    for e in encoded:
        tgt.append(list(e) + [0] * (max_len - len(e)))
        ones = len(e) - 1
        weights.append([1.0] * ones + [0.0] * (max_len - ones))
    tgt_input = np.array(tgt, dtype=np.int64).T
    tgt_output = np.roll(tgt_input, -1, 0).T.copy()
    tgt_output[:, -1] = 0
    mask = np.array(weights) == 0
    return tgt_input, tgt_output, mask


def augment(img_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light photometric noise typical of phone photos of thermal paper (geometry is varied by the padding)."""
    import cv2
    out = img_rgb.astype(np.float32)
    if rng.random() < 0.5:                                    # brightness / contrast
        out = out * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    if rng.random() < 0.2:                                    # slight blur
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.2:                                    # sensor noise
        out = out + rng.normal(0.0, rng.uniform(3, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def cer_summary(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    """Corpus CER (total edits / total reference characters, after ``normalize_text``) and exact-match rate."""
    edits = chars = exact = 0
    for r, h in zip(refs, hyps):
        rn, hn = normalize_text(r), normalize_text(h)
        edits += edit_distance(rn, hn)
        chars += len(rn)
        exact += rn == hn
    n = len(refs)
    return {"n": n, "cer": edits / chars if chars else 0.0, "exact": exact / n if n else 0.0}


def per_receipt_edits(rows: Sequence[Dict[str, Any]], hyps: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """``img_id -> (character edits, reference characters)`` (the input of ``ocr_ablation.paired_cer_difference``)."""
    acc: Dict[str, List[int]] = {}
    for r, h in zip(rows, hyps):
        rn, hn = normalize_text(r["text"]), normalize_text(h)
        a = acc.setdefault(r["img_id"], [0, 0])
        a[0] += edit_distance(rn, hn)
        a[1] += len(rn)
    return {k: (v[0], v[1]) for k, v in acc.items()}


def lr_factor(step: int, total: int, warmup: float = 0.05) -> float:
    """Linear warm-up then cosine decay to 5 % of the peak (never errors past ``total``)."""
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    p = min(1.0, (step - w) / max(1, total - w))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------------- model side
def load_vietocr(device: str, weights: Optional[str] = None, backbone: str = "vgg_transformer") -> Any:
    """VietOCR ``Predictor`` (model + vocab + config) with the pretrained weights or a fine-tuned state dict."""
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    cfg = Cfg.load_config_from_name(backbone)
    cfg["device"] = device
    cfg["predictor"]["beamsearch"] = False
    if weights:
        cfg["weights"] = weights
    return Predictor(cfg)


def _load_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].copy()


class VietOCRFineTuner:
    """Holds the predictor and the crops; ``fit`` trains, ``read`` / ``evaluate`` read with inference padding."""

    def __init__(self, predictor: Any, batch_size: int = 32, seed: int = 42) -> None:
        self.predictor = predictor
        self.model = predictor.model
        self.vocab = predictor.vocab
        self.device = predictor.device
        ds = predictor.config["dataset"]
        self.height, self.min_w, self.max_w = ds["image_height"], ds["image_min_width"], ds["image_max_width"]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, np.ndarray] = {}

    # -- data ---------------------------------------------------------------------
    def crop(self, row: Dict[str, Any]) -> np.ndarray:
        if row["crop"] not in self._cache:
            self._cache[row["crop"]] = _load_rgb(row["crop"])
        return self._cache[row["crop"]]

    def view(self, row: Dict[str, Any], pad: Tuple[int, int] = INFER_PAD) -> np.ndarray:
        return np.ascontiguousarray(padded_view(self.crop(row), row["inner"], pad[0], pad[1]))

    def _batch_tensor(self, views: Sequence[np.ndarray]) -> Any:
        import torch
        from PIL import Image
        from vietocr.tool.translate import process_image
        arr = np.stack([process_image(Image.fromarray(v), self.height, self.min_w, self.max_w) for v in views])
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    # -- reading ------------------------------------------------------------------
    def read(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        """Greedy reading of every row with the inference padding (same decoding as ``Predictor.predict_batch``)."""
        import torch
        from vietocr.tool.translate import translate
        views = [self.view(r) for r in rows]
        widths = [resized_width(v.shape[1], v.shape[0], self.height, self.min_w, self.max_w) for v in views]
        out: List[str] = [""] * len(rows)
        self.model.eval()
        with torch.no_grad():
            for batch in width_buckets(widths, 64):
                s, _ = translate(self._batch_tensor([views[i] for i in batch]), self.model)
                for i, text in zip(batch, self.vocab.batch_decode(s.tolist())):
                    out[i] = text
        return out

    def evaluate(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """CER overall, per pair kind and per field label, plus the readings (for paired comparisons)."""
        hyps = self.read(rows)
        refs = [r["text"] for r in rows]
        res: Dict[str, Any] = {"all": cer_summary(refs, hyps), "by_kind": {}, "by_label": {}}
        for key, out_key in (("kind", "by_kind"), ("label", "by_label")):
            for value in sorted({r[key] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r[key] == value]
                res[out_key][value] = cer_summary([refs[i] for i in idx], [hyps[i] for i in idx])
        res["hyps"] = hyps
        return res

    # -- training -----------------------------------------------------------------
    def fit(self, train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]], out_path: str,
            epochs: int = 12, lr: float = 1e-4, patience: int = 3, select_kind: str = "region",
            log: Any = print) -> Dict[str, Any]:
        """Train, evaluate on ``val_rows`` after every epoch and keep the state dict with the lowest CER.

        The selection CER is computed on the ``select_kind`` pairs of validation (``region`` = the crops the pipeline
        reads). Epoch 0 is the pretrained model, so the history starts with the baseline.
        """
        import torch
        from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss

        sel_rows = [r for r in val_rows if r["kind"] == select_kind] or list(val_rows)
        crit = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
        steps_per_epoch = math.ceil(len(train_rows) / self.batch_size) + 60      # upper bound (partial buckets)
        total = steps_per_epoch * epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total))
        encoded = [self.vocab.encode(r["text"]) for r in train_rows]

        base = cer_summary([r["text"] for r in sel_rows], self.read(sel_rows))
        history = [{"epoch": 0, "val_cer": base["cer"], "val_exact": base["exact"], "train_loss": None}]
        log(f"[ft] epoch 0 (pretrained) val {select_kind} CER={base['cer']:.4f} exact={base['exact']:.3f} "
            f"| train pairs={len(train_rows)} val pairs={len(sel_rows)}")
        best, bad = base["cer"], 0
        torch.save(self.model.state_dict(), out_path)                            # the pretrained model is the floor
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            pads = [(int(self.rng.integers(PAD_X_RANGE[0], PAD_X_RANGE[1] + 1)),
                     int(self.rng.integers(PAD_Y_RANGE[0], PAD_Y_RANGE[1] + 1))) for _ in train_rows]
            views = [augment(self.view(r, p), self.rng) for r, p in zip(train_rows, pads)]
            widths = [resized_width(v.shape[1], v.shape[0], self.height, self.min_w, self.max_w) for v in views]
            batches = width_buckets(widths, self.batch_size, self.rng)
            loss_sum = 0.0
            for k, batch in enumerate(batches):
                img = self._batch_tensor([views[i] for i in batch])
                ti, to, mask = target_tensors([encoded[i] for i in batch])
                ti, to = torch.as_tensor(ti, device=self.device), torch.as_tensor(to, device=self.device)
                mask = torch.as_tensor(mask, device=self.device)
                out = self.model(img, ti, tgt_key_padding_mask=mask)
                loss = crit(out.reshape(-1, out.size(2)), to.reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_sum += loss.item()
                if (k + 1) % max(1, len(batches) // 5) == 0:
                    log(f"  epoch {epoch}/{epochs} {100 * (k + 1) // len(batches)}% loss={loss_sum / (k + 1):.4f} "
                        f"{time.time() - t0:.0f}s")
            ev = cer_summary([r["text"] for r in sel_rows], self.read(sel_rows))
            improved = ev["cer"] < best - 1e-6
            history.append({"epoch": epoch, "val_cer": ev["cer"], "val_exact": ev["exact"],
                            "train_loss": loss_sum / max(1, len(batches)), "seconds": round(time.time() - t0, 1)})
            log(f"[ft] epoch {epoch} train_loss={history[-1]['train_loss']:.4f} val {select_kind} CER={ev['cer']:.4f} "
                f"exact={ev['exact']:.3f} {time.time() - t0:.0f}s{'  *best*' if improved else ''}")
            if improved:
                best, bad = ev["cer"], 0
                torch.save(self.model.state_dict(), out_path)
            else:
                bad += 1
                if bad >= patience:
                    log(f"[ft] early stop: no CER improvement for {patience} epochs")
                    break
        self.model.load_state_dict(torch.load(out_path, map_location=self.device))
        best_epoch = min(history, key=lambda h: h["val_cer"])["epoch"]
        return {"history": history, "best_val_cer": best, "best_epoch": best_epoch, "weights": out_path}
