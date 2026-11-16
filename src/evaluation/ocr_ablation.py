"""RQ1 experiment: how image preprocessing changes VietOCR's text recognition (CER / WER on annotated regions).

Why annotated regions: MC-OCR gives the text of every SELLER / ADDRESS / TIMESTAMP / TOTAL_COST region, so each region
can be cropped from the image and read by VietOCR, and the reading compared with the annotation. This isolates the
recognition step from detection (a missed or merged CRAFT box cannot hide the effect of preprocessing).

For every receipt the original photo is optionally degraded (blur, darkening, noise; fixed parameters, not tuned),
preprocessed with one of the configurations in ``CONFIGS`` and every annotated polygon is mapped into the processed
image with the exact transform of ``ImageGeometry`` (resize then rotation by the estimated skew). Everything except
the recognizer is plain Python / OpenCV; the recognizer is injected, so the logic is unit-tested with a fake one.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.evaluation.metrics import edit_distance, normalize_text
from src.preprocessing.annotation_parser import load_and_validate_mcocr_csv
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.token_dataset import ImageGeometry, frame_status, map_annotation_regions, read_split_manifest

# Preprocessing configurations: resizing to ``target_max_dim`` is always applied; the rest is switched on or off.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "none": {"enhance": False, "max_skew": 0.0},        # resize only
    "deskew": {"enhance": False, "max_skew": 15.0},     # + rotation by the estimated skew
    "contrast": {"enhance": True, "max_skew": 0.0},     # + CLAHE and bilateral filter
    "full": {"enhance": True, "max_skew": 15.0},        # both (the setting of the main pipeline)
}
DEGRADATIONS = ("none", "blur", "dark", "noise")
QUALITY_GROUPS = ("low", "mid", "high")


# ---------------------------------------------------------------------------------- pure helpers
def degrade(image_bgr: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Synthetic degradation of the original photo (fixed parameters)."""
    if kind == "none":
        return image_bgr.copy()
    if kind == "blur":
        return cv2.GaussianBlur(image_bgr, (0, 0), 2.5)
    if kind == "dark":
        return np.clip(image_bgr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        noisy = image_bgr.astype(np.float32) + rng.normal(0.0, 18.0, image_bgr.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown degradation: {kind}")


def quality_groups(qualities: Dict[str, float]) -> Dict[str, str]:
    """``img_id -> low | mid | high`` by thirds of ``anno_image_quality`` (ties broken by id, so it is deterministic)."""
    ordered = sorted(qualities, key=lambda k: (qualities[k], k))
    n = len(ordered)
    out: Dict[str, str] = {}
    for i, img_id in enumerate(ordered):
        out[img_id] = QUALITY_GROUPS[0] if i < n / 3 else (QUALITY_GROUPS[1] if i < 2 * n / 3 else QUALITY_GROUPS[2])
    return out


def crop_regions(image_bgr: np.ndarray, boxes: Sequence[Sequence[float]], pad_x: int = 3, pad_y: int = 2
                 ) -> Tuple[List[np.ndarray], List[int]]:
    """RGB crops of ``xyxy`` boxes with the padding the OCR stage uses; tiny boxes are skipped.

    Returns ``(crops, kept)`` where ``kept`` holds the index of the box each crop came from.
    """
    h, w = image_bgr.shape[:2]
    rgb = image_bgr[:, :, ::-1]
    crops: List[np.ndarray] = []
    kept: List[int] = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        x0, y0 = max(0, int(np.floor(x0)) - pad_x), max(0, int(np.floor(y0)) - pad_y)
        x1, y1 = min(w, int(np.ceil(x1)) + pad_x), min(h, int(np.ceil(y1)) + pad_y)
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            continue
        crops.append(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        kept.append(i)
    return crops, kept


def _error_counts(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """``(char edits, char length, word edits, word length)`` after normalisation (diacritics are kept)."""
    r, h = normalize_text(ref), normalize_text(hyp)
    return edit_distance(r, h), len(r), edit_distance(r.split(), h.split()), len(r.split())


def _rate(rows: Sequence[Dict[str, Any]], edits: str, length: str) -> Optional[float]:
    total = sum(r[length] for r in rows)
    return sum(r[edits] for r in rows) / total if total else None


# ---------------------------------------------------------------------------------- loading
def load_items(csv_path: str, split_manifest: str, split: str, image_dir: str,
               max_receipts: Optional[int] = None) -> List[Dict[str, Any]]:
    """Receipts of one split with their annotation instances, image quality and original image path."""
    records, _ = load_and_validate_mcocr_csv(csv_path)
    by_id = {r.img_id: r for r in records if r.is_valid and r.instances}
    ids = sorted(i for i, s in read_split_manifest(split_manifest).items() if s == split and i in by_id)
    if max_receipts:
        ids = ids[:max_receipts]
    return [{"img_id": i, "path": os.path.join(image_dir, i), "quality": by_id[i].image_quality,
             "instances": by_id[i].instances} for i in ids]


# ---------------------------------------------------------------------------------- the experiment
def run_ablation(items: Sequence[Dict[str, Any]], recognizer: Any, target_max_dim: int = 1280,
                 configs: Optional[Dict[str, Dict[str, Any]]] = None, degradations: Sequence[str] = ("none",),
                 batch_size: int = 64, seed: int = 0) -> Dict[str, Any]:
    """Read every annotated region under every (degradation, configuration) and compare with the annotation.

    ``recognizer`` needs ``recognize_batch(list of RGB crops) -> list of objects with .text``.
    Returns ``{"rows": [...], "skipped": {...}, "groups": {img_id: group}}``.
    """
    configs = configs or CONFIGS
    preprocessors = {name: ImagePreprocessor(target_max_dim=target_max_dim, **kw) for name, kw in configs.items()}
    groups = quality_groups({it["img_id"]: it["quality"] for it in items})
    skipped = {"unreadable": 0, "frame_mismatch": 0}
    rows: List[Dict[str, Any]] = []

    for deg_index, deg in enumerate(degradations):
        entries: Dict[str, List[Tuple[Dict[str, Any], Any, np.ndarray]]] = {name: [] for name in configs}
        for item in items:
            orig = cv2.imread(item["path"])
            if orig is None:
                if deg_index == 0:
                    skipped["unreadable"] += 1
                continue
            oh, ow = orig.shape[:2]
            if any(frame_status((i.width, i.height), (ow, oh)) != "ok" for i in item["instances"]):
                if deg_index == 0:
                    skipped["frame_mismatch"] += 1
                continue
            degraded = degrade(orig, deg, seed)
            for name, pre in preprocessors.items():
                res = pre.process_array(degraded)
                ph, pw = res.image.shape[:2]
                regions = map_annotation_regions(item["instances"], ImageGeometry(ow, oh, pw, ph, res.skew_angle))
                crops, kept = crop_regions(res.image, [r.bbox for r in regions])
                for j, crop in zip(kept, crops):
                    entries[name].append((item, regions[j], crop))

        for name, items_crops in entries.items():
            hyps: List[str] = []
            for i in range(0, len(items_crops), batch_size):
                batch = [c for _, _, c in items_crops[i:i + batch_size]]
                hyps.extend((r.text or "") for r in recognizer.recognize_batch(batch))
            for (item, region, _), hyp in zip(items_crops, hyps):
                ce, cl, we, wl = _error_counts(region.text, hyp)
                if cl == 0:
                    continue                     # an empty reference has no characters to get wrong
                rows.append({"degradation": deg, "config": name, "img_id": item["img_id"],
                             "group": groups[item["img_id"]], "label": region.label, "ref": region.text, "hyp": hyp,
                             "char_edits": ce, "char_len": cl, "word_edits": we, "word_len": wl})
    return {"rows": rows, "skipped": skipped, "groups": groups}


# ---------------------------------------------------------------------------------- summaries
def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{"<degradation>/<config>": {regions, cer, wer, by_group, by_label}}`` (corpus-level rates)."""
    keys = sorted({(r["degradation"], r["config"]) for r in rows})
    out: Dict[str, Dict[str, Any]] = {}
    for deg, cfg in keys:
        sub = [r for r in rows if r["degradation"] == deg and r["config"] == cfg]
        out[f"{deg}/{cfg}"] = {
            "regions": len(sub),
            "cer": _rate(sub, "char_edits", "char_len"),
            "wer": _rate(sub, "word_edits", "word_len"),
            "by_group": {g: {"regions": len([r for r in sub if r["group"] == g]),
                             "cer": _rate([r for r in sub if r["group"] == g], "char_edits", "char_len")}
                         for g in QUALITY_GROUPS},
            "by_label": {lab: _rate([r for r in sub if r["label"] == lab], "char_edits", "char_len")
                         for lab in sorted({r["label"] for r in sub})},
        }
    return out


def per_receipt_counts(rows: Sequence[Dict[str, Any]], degradation: str, config: str) -> Dict[str, Tuple[int, int]]:
    """``img_id -> (character edits, reference characters)`` for one (degradation, configuration)."""
    acc: Dict[str, List[int]] = {}
    for r in rows:
        if r["degradation"] == degradation and r["config"] == config:
            a = acc.setdefault(r["img_id"], [0, 0])
            a[0] += r["char_edits"]
            a[1] += r["char_len"]
    return {k: (v[0], v[1]) for k, v in acc.items()}


def paired_cer_difference(base: Dict[str, Tuple[int, int]], other: Dict[str, Tuple[int, int]], n_boot: int = 1000,
                          seed: int = 0) -> Dict[str, float]:
    """CER(other) - CER(base) over the same receipts, with a 95 % bootstrap interval (negative = other is better).

    Receipts are resampled with replacement and both systems are scored on the same resample.
    """
    ids = sorted(set(base) & set(other))
    if not ids:
        return {"n": 0, "diff": 0.0, "low": 0.0, "high": 0.0, "share_better": 0.0}
    eb = np.array([base[i][0] for i in ids], dtype=np.float64)
    lb = np.array([base[i][1] for i in ids], dtype=np.float64)
    eo = np.array([other[i][0] for i in ids], dtype=np.float64)
    lo = np.array([other[i][1] for i in ids], dtype=np.float64)
    observed = eo.sum() / max(lo.sum(), 1.0) - eb.sum() / max(lb.sum(), 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    diffs = (eo[idx].sum(axis=1) / np.maximum(lo[idx].sum(axis=1), 1.0)
             - eb[idx].sum(axis=1) / np.maximum(lb[idx].sum(axis=1), 1.0))
    return {"n": len(ids), "diff": float(observed), "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5)), "share_better": float((diffs < 0).mean())}


def build_report(result: Dict[str, Any], degradations: Sequence[str], configs: Sequence[str],
                 n_boot: int = 1000) -> Dict[str, Any]:
    rows = result["rows"]
    paired: Dict[str, Dict[str, float]] = {}
    for deg in degradations:
        base = per_receipt_counts(rows, deg, "none")
        for cfg in configs:
            if cfg != "none":
                paired[f"{deg}/{cfg} vs {deg}/none"] = paired_cer_difference(base, per_receipt_counts(rows, deg, cfg), n_boot)
    return {"skipped": result["skipped"], "n_receipts": len(result["groups"]), "summary": summarize(rows),
            "paired_vs_none": paired,
            "note": "CER and WER are corpus-level (total edits / total reference length) on annotated regions read by "
                    "VietOCR; diacritics are kept; 'none' = resize only."}


def format_markdown(report: Dict[str, Any]) -> str:
    def pct(v: Optional[float]) -> str:
        return "-" if v is None else f"{100 * v:.1f}"

    lines = ["| Degradation / config | Regions | CER % | WER % | CER low % | CER mid % | CER high % |",
             "|---|---|---|---|---|---|---|"]
    for key, s in report["summary"].items():
        g = s["by_group"]
        lines.append(f"| {key} | {s['regions']} | {pct(s['cer'])} | {pct(s['wer'])} | {pct(g['low']['cer'])} | "
                     f"{pct(g['mid']['cer'])} | {pct(g['high']['cer'])} |")
    lines += ["", "Paired difference in CER against 'none' (percentage points; negative = better; 95 % bootstrap interval):", "",
              "| Comparison | Receipts | Difference | Interval | Share of resamples better |", "|---|---|---|---|---|"]
    for key, d in report["paired_vs_none"].items():
        lines.append(f"| {key} | {d['n']} | {100 * d['diff']:+.2f} | [{100 * d['low']:+.2f}, {100 * d['high']:+.2f}] | "
                     f"{d['share_better']:.2f} |")
    return "\n".join(lines) + "\n"
