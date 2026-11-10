"""Build the BIO token-classification dataset from OCR output and MC-OCR annotations.

Pure Python (no torch, no OpenCV) so every function can be unit-tested anywhere.

Coordinates
-----------
MC-OCR polygons live in the annotation frame (``width`` x ``height`` stored with each polygon). The OCR JSON
written by ``ReceiptPipeline`` lives in the *processed* image: the original resized, then rotated by
``skew_angle`` around ``(w // 2, h // 2)`` (OpenCV ``getRotationMatrix2D`` + ``warpAffine``). ``ImageGeometry``
reproduces exactly that transform for the annotation polygons, so labels and OCR words share one frame.

Labelling
---------
Each OCR region is split into words with the same function the LayoutXLM extractor uses
(``split_box_by_tokens``), so training words equal inference words. A word gets the field of the annotation
region that contains most of its box (at least ``min_containment``); everything else is ``O``. Consecutive words
with the same field form one entity (``B-`` then ``I-``).
"""
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.bbox_utils import contains_ratio, normalize_bbox
from src.evaluation.metrics import cer, group_gt_fields, normalize_text
from src.extraction.schema import ANNOTATED_FIELDS
from src.features.layoutxlm_extractor import split_box_by_tokens
from src.preprocessing.annotation_parser import load_and_validate_mcocr_csv

# ---------------------------------------------------------------------------------- label set
LABELS: List[str] = ["O"] + [f"{p}-{f}" for f in ANNOTATED_FIELDS for p in ("B", "I")]
LABEL2ID: Dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}

DEFAULT_MIN_CONTAINMENT = 0.5      # share of a word box that must lie inside an annotation box
DEFAULT_FRAME_TOL = 0.02           # allowed relative aspect-ratio difference annotation vs image
MATCHED_CER = 0.5                  # a region counts as "recovered" when CER(gt, assigned words) <= this

Point = Tuple[float, float]


# ---------------------------------------------------------------------------------- geometry
def _pairs(nums: Sequence[float]) -> List[Point]:
    return [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]


def _collect_points(obj: Any, out: List[Point]) -> None:
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            out.extend(_pairs(obj))
        else:
            for v in obj:
                _collect_points(v, out)


def polygon_points(segmentation: Any) -> List[Point]:
    """All vertices of a segmentation: a flat ``[x1, y1, ...]`` ring, a list of rings, or ``[[x, y], ...]``."""
    pts: List[Point] = []
    _collect_points(segmentation, pts)
    return pts


def rotation_matrix(angle_deg: float, center: Tuple[float, float]) -> Tuple[Tuple[float, float, float],
                                                                             Tuple[float, float, float]]:
    """Same matrix as ``cv2.getRotationMatrix2D(center, angle_deg, 1.0)``.

    ``warpAffine`` without ``WARP_INVERSE_MAP`` sends a source point ``p`` to ``M @ [x, y, 1]``.
    """
    rad = math.radians(angle_deg)
    alpha, beta = math.cos(rad), math.sin(rad)
    cx, cy = center
    return ((alpha, beta, (1.0 - alpha) * cx - beta * cy),
            (-beta, alpha, beta * cx + (1.0 - alpha) * cy))


def frame_status(ref_size: Tuple[float, float], orig_size: Tuple[float, float],
                 tol: float = DEFAULT_FRAME_TOL) -> str:
    """Is the annotation frame consistent with the original image?

    ``ok``: same aspect ratio (scale differences are fine). ``transposed``: width and height are swapped
    (typically an EXIF-rotated photo). ``aspect_mismatch``: anything else, including missing sizes.
    """
    rw, rh = ref_size
    ow, oh = orig_size
    if min(rw, rh, ow, oh) <= 0:
        return "aspect_mismatch"
    ar_ref, ar_orig = rw / rh, ow / oh
    if abs(ar_ref / ar_orig - 1.0) <= tol:
        return "ok"
    if abs(ar_ref * ar_orig - 1.0) <= tol:
        return "transposed"
    return "aspect_mismatch"


class ImageGeometry:
    """Original size, processed size and skew angle of one image (all from the OCR JSON)."""

    def __init__(self, orig_w: int, orig_h: int, proc_w: int, proc_h: int, angle: float = 0.0) -> None:
        self.orig_w, self.orig_h = int(orig_w), int(orig_h)
        self.proc_w, self.proc_h = int(proc_w), int(proc_h)
        self.angle = float(angle)

    @classmethod
    def from_ocr_meta(cls, meta: Dict[str, Any]) -> "ImageGeometry":
        oh, ow = meta["orig_shape"][:2]
        ph, pw = meta["processed_shape"][:2]
        return cls(ow, oh, pw, ph, meta.get("skew_angle", 0.0) or 0.0)

    def map_points(self, points: Iterable[Point], ref_size: Optional[Tuple[float, float]] = None) -> List[Point]:
        """Annotation-frame points -> processed-image points (resize, then rotation about the centre)."""
        rw, rh = ref_size if ref_size else (self.orig_w, self.orig_h)
        sx, sy = self.proc_w / float(rw), self.proc_h / float(rh)
        (a, b, tx), (c, d, ty) = rotation_matrix(self.angle, (self.proc_w // 2, self.proc_h // 2))
        out: List[Point] = []
        for x, y in points:
            x, y = x * sx, y * sy
            out.append((a * x + b * y + tx, c * x + d * y + ty))
        return out

    def map_bbox(self, points: Iterable[Point], ref_size: Optional[Tuple[float, float]] = None) -> List[float]:
        """Axis-aligned ``xyxy`` box of the mapped points, clipped to the processed image."""
        mapped = self.map_points(points, ref_size)
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        return [max(0.0, min(xs)), max(0.0, min(ys)),
                min(float(self.proc_w), max(xs)), min(float(self.proc_h), max(ys))]


# ---------------------------------------------------------------------------------- annotations
class GtRegion:
    """One annotated region mapped into the processed image."""

    def __init__(self, label: str, text: str, bbox: List[float]) -> None:
        self.label = label
        self.text = text
        self.bbox = bbox

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


def map_annotation_regions(instances: Iterable[Any], geom: ImageGeometry) -> List[GtRegion]:
    """Map every annotated instance (``AnnotationInstance``) into the processed image.

    Instances without a usable polygon fall back to their ``xywh`` box.
    """
    regions: List[GtRegion] = []
    for inst in instances:
        if inst.label not in ANNOTATED_FIELDS:
            continue
        pts = polygon_points(inst.segmentation)
        if len(pts) < 3 and inst.raw_bbox and len(inst.raw_bbox) == 4:
            x, y, w, h = inst.raw_bbox
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if len(pts) < 3:
            continue
        ref = (inst.width, inst.height) if inst.width > 0 and inst.height > 0 else None
        regions.append(GtRegion(inst.label, inst.text, geom.map_bbox(pts, ref)))
    return regions


# ---------------------------------------------------------------------------------- words and labels
def items_to_words(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OCR items -> words with their (processed-pixel, xyxy) boxes, in reading order.

    Identical to ``build_words_and_boxes`` of the LayoutXLM extractor, plus the index of the source item.
    """
    words: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        tokens = (item.get("text") or "").split()
        if not tokens:
            continue
        for tok, box in zip(tokens, split_box_by_tokens(item["bbox"], tokens)):
            words.append({"text": tok, "bbox": box, "item": idx})
    return words


def _at_least_one_pixel(box: Sequence[float]) -> List[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, max(x1, x0 + 1.0), max(y1, y0 + 1.0)]


def assign_word_fields(words: Sequence[Dict[str, Any]], regions: Sequence[GtRegion],
                       min_containment: float = DEFAULT_MIN_CONTAINMENT
                       ) -> Tuple[List[Optional[str]], List[Optional[int]]]:
    """Field and region index for every word (``None`` when the word is outside all annotated regions).

    The region containing the largest share of the word box wins; ties go to the smaller region.
    """
    fields: List[Optional[str]] = []
    owners: List[Optional[int]] = []
    for w in words:
        box = _at_least_one_pixel(w["bbox"])
        best_key, best_idx = None, None
        for ri, r in enumerate(regions):
            share = contains_ratio(box, r.bbox)
            if share >= min_containment:
                key = (share, -r.area)
                if best_key is None or key > best_key:
                    best_key, best_idx = key, ri
        owners.append(best_idx)
        fields.append(regions[best_idx].label if best_idx is not None else None)
    return fields, owners


def bio_tags(fields: Sequence[Optional[str]]) -> List[str]:
    """``B-X`` at the start of a run of the same field, ``I-X`` inside it, ``O`` for unlabelled words."""
    tags: List[str] = []
    prev: Optional[str] = None
    for f in fields:
        if f is None:
            tags.append("O")
        else:
            tags.append(("I-" if f == prev else "B-") + f)
        prev = f
    return tags


# ---------------------------------------------------------------------------------- one image
def build_token_record(file_id: str, meta: Dict[str, Any], annotation: Any,
                       min_containment: float = DEFAULT_MIN_CONTAINMENT,
                       frame_tol: float = DEFAULT_FRAME_TOL) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Token record + diagnostics for one image. The record is ``None`` when the image cannot be labelled safely.

    ``meta`` is the OCR JSON, ``annotation`` an ``MCOCRRecord``. Diagnostics always carry ``status`` in
    {``ok``, ``no_ocr``, ``frame_mismatch``}.
    """
    diag: Dict[str, Any] = {"file_id": file_id, "status": "ok", "regions": [], "n_words": 0, "words_per_tag": {}}
    items = meta.get("ocr_items") or []
    words = items_to_words(items)
    if not words:
        diag["status"] = "no_ocr"
        return None, diag

    geom = ImageGeometry.from_ocr_meta(meta)
    for inst in annotation.instances:
        status = frame_status((inst.width, inst.height), (geom.orig_w, geom.orig_h), frame_tol)
        if status != "ok":
            diag.update(status="frame_mismatch", frame_detail=status,
                        annotation_size=[inst.width, inst.height], orig_size=[geom.orig_w, geom.orig_h])
            return None, diag

    regions = map_annotation_regions(annotation.instances, geom)
    fields, owners = assign_word_fields(words, regions, min_containment)
    tags = bio_tags(fields)

    for ri, r in enumerate(regions):
        assigned = [words[i]["text"] for i, o in enumerate(owners) if o == ri]
        entry: Dict[str, Any] = {"label": r.label, "covered": bool(assigned), "cer": None}
        if assigned:
            entry["cer"] = cer(normalize_text(r.text), normalize_text(" ".join(assigned)))
        diag["regions"].append(entry)

    counts: Dict[str, int] = {}
    for t in tags:
        counts[t] = counts.get(t, 0) + 1
    diag["n_words"] = len(words)
    diag["words_per_tag"] = counts

    record = {
        "file_id": file_id,
        "img_id": annotation.img_id,
        "image": f"{file_id}.jpg",
        "width": geom.proc_w,
        "height": geom.proc_h,
        "quality": annotation.image_quality,
        "words": [w["text"] for w in words],
        "boxes": [normalize_bbox(w["bbox"], geom.proc_w, geom.proc_h, scale=1000, is_xywh=False) for w in words],
        "labels": tags,
        "gt": group_gt_fields(annotation.instances),
    }
    return record, diag


# ---------------------------------------------------------------------------------- reports
def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return float(sorted_vals[idx])


def summarize_diagnostics(diags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-image diagnostics.

    ``coverage`` = annotated regions that received at least one word (an upper bound for recall of any
    model trained on these labels); ``recovered`` = regions whose assigned words are close to the annotated
    text (CER <= ``MATCHED_CER``).
    """
    status_counts: Dict[str, int] = {}
    per_label: Dict[str, Dict[str, Any]] = {f: {"regions": 0, "covered": 0, "recovered": 0, "cer_sum": 0.0}
                                            for f in ANNOTATED_FIELDS}
    tag_counts: Dict[str, int] = {}
    n_words: List[int] = []
    for d in diags:
        status_counts[d["status"]] = status_counts.get(d["status"], 0) + 1
        if d["status"] != "ok":
            continue
        n_words.append(d["n_words"])
        for tag, n in d["words_per_tag"].items():
            tag_counts[tag] = tag_counts.get(tag, 0) + n
        for r in d["regions"]:
            s = per_label[r["label"]]
            s["regions"] += 1
            if r["covered"]:
                s["covered"] += 1
                s["cer_sum"] += r["cer"]
                if r["cer"] <= MATCHED_CER:
                    s["recovered"] += 1

    labels_out: Dict[str, Dict[str, Any]] = {}
    for f, s in per_label.items():
        n, c = s["regions"], s["covered"]
        labels_out[f] = {
            "regions": n,
            "coverage": c / n if n else 0.0,
            "recovered": s["recovered"] / n if n else 0.0,
            "mean_cer_of_covered": s["cer_sum"] / c if c else 0.0,
        }
    n_words.sort()
    return {
        "images_by_status": status_counts,
        "per_label": labels_out,
        "words_per_tag": dict(sorted(tag_counts.items())),
        "words_per_image": {
            "mean": sum(n_words) / len(n_words) if n_words else 0.0,
            "p95": _percentile(n_words, 0.95),
            "max": float(n_words[-1]) if n_words else 0.0,
        },
    }


# ---------------------------------------------------------------------------------- whole dataset
def read_split_manifest(path: str) -> Dict[str, str]:
    """``img_id -> split`` from ``mcocr_manifest_split.csv`` (one row per region, so deduplicate)."""
    import csv
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["img_id"]] = row["split"]
    return out


def build_dataset(csv_path: str, ocr_dir: str, split_manifest: str, out_dir: str,
                  min_containment: float = DEFAULT_MIN_CONTAINMENT) -> Dict[str, Any]:
    """Write ``<split>.jsonl`` files and ``token_dataset_report.json`` into ``out_dir``; return the report."""
    records, _ = load_and_validate_mcocr_csv(csv_path)
    by_id = {r.img_id: r for r in records if r.is_valid and r.instances}
    splits = read_split_manifest(split_manifest)
    os.makedirs(out_dir, exist_ok=True)

    writers: Dict[str, Any] = {}
    diags_by_split: Dict[str, List[Dict[str, Any]]] = {}
    missing_ocr: List[str] = []
    missing_annotation: List[str] = []
    skipped: List[Dict[str, Any]] = []
    written: Dict[str, int] = {}
    try:
        for img_id in sorted(splits):
            split = splits[img_id]
            file_id = os.path.splitext(img_id)[0]
            ann = by_id.get(img_id)
            if ann is None:
                missing_annotation.append(img_id)
                continue
            json_path = os.path.join(ocr_dir, f"{file_id}.json")
            if not os.path.exists(json_path):
                missing_ocr.append(img_id)
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            record, diag = build_token_record(file_id, meta, ann, min_containment)
            diags_by_split.setdefault(split, []).append(diag)
            if record is None:
                skipped.append({k: diag[k] for k in diag if k not in ("regions", "words_per_tag")})
                continue
            if split not in writers:
                writers[split] = open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8")
            writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            written[split] = written.get(split, 0) + 1
    finally:
        for w in writers.values():
            w.close()

    report = {
        "min_containment": min_containment,
        "labels": LABELS,
        "images_in_split_manifest": len(splits),
        "records_written": written,
        "missing_ocr": missing_ocr,
        "missing_annotation": missing_annotation,
        "skipped": skipped,
        "by_split": {s: summarize_diagnostics(d) for s, d in diags_by_split.items()},
        "all": summarize_diagnostics([d for ds in diags_by_split.values() for d in ds]),
    }
    with open(os.path.join(out_dir, "token_dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
