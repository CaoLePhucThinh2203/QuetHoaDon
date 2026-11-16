"""Training pairs (image crop, annotated text) for fine-tuning VietOCR on MC-OCR receipts (pure Python + OpenCV).

Why: 117 of the 160 wrong fields on validation are OCR text errors, and the pretrained VietOCR has never seen thermal
receipts. MC-OCR annotates the text of every SELLER / ADDRESS / TIMESTAMP / TOTAL_COST line, so the recogniser can be
taught exactly the text the extraction needs.

Two kinds of pairs are made from every annotated line of a receipt:

* ``line``: the whole annotated line (axis-aligned box of the polygon). The label is exact.
* ``region``: the CRAFT boxes that lie inside the line (the crops the pipeline really reads). Their label is the part of
  the line text that the box covers, found by aligning the current OCR reading of the boxes with the annotated text
  (``align_segments``). Pairs whose alignment is doubtful are skipped (see ``RegionPairConfig``).

Crops come from the *processed* image written by the OCR run (same resize, deskew and enhancement as inference) and are
saved with a margin, so training can jitter the padding around the box; evaluation always uses the inference padding.
Only the splits asked for are written (fine-tuning uses ``train``; ``validation`` is for model selection and the gate).
"""
import json
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.bbox_utils import contains_ratio
from src.evaluation.metrics import cer, normalize_text
from src.preprocessing.annotation_parser import load_and_validate_mcocr_csv
from src.preprocessing.token_dataset import ImageGeometry, frame_status, map_annotation_regions, read_split_manifest

INFER_PAD = (3, 2)           # padding (x, y) the recogniser uses around a CRAFT box (BaseTextRecognizer.recognize_regions)
CROP_MARGIN = 8              # pixels saved around every box so training can vary the padding


@dataclass
class RegionPairConfig:
    min_containment: float = 0.8   # share of a CRAFT box that must lie inside the annotated line
    max_line_cer: float = 0.5      # skip a line whose joined OCR reading is further than this from the annotation
    max_segment_cer: float = 0.6   # skip one box whose aligned label is further than this from its OCR reading


# ---------------------------------------------------------------------------------- alignment
def _alignment_map(hyp: str, gold: str) -> List[int]:
    """``m[i]`` = position in ``gold`` that corresponds to position ``i`` of ``hyp`` (``len(m) == len(hyp) + 1``).

    Levenshtein alignment on lower-cased characters; for an inserted hyp character the gold position does not move.
    """
    h, g = hyp.lower(), gold.lower()
    n, m = len(h), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (h[i - 1] != g[j - 1]))
    pos = [0] * (n + 1)
    i, j = n, m
    pos[n] = m
    while i > 0:
        if j > 0 and dp[i][j] == dp[i - 1][j - 1] + (h[i - 1] != g[j - 1]):
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
            continue
        pos[i] = j
    return pos


def _snap_to_space(gold: str, cut: int, reach: int = 2) -> int:
    """Move a cut position onto the nearest space of ``gold`` within ``reach`` characters (keep it otherwise)."""
    if cut <= 0 or cut >= len(gold) or gold[cut] == " " or gold[cut - 1] == " ":
        return cut
    for d in range(1, reach + 1):
        for c in (cut - d, cut + d):
            if 0 < c < len(gold) and gold[c] == " ":
                return c
    return cut


def align_segments(parts: Sequence[str], gold: str) -> List[str]:
    """Split ``gold`` into one piece per OCR part (parts are read left to right and joined with one space).

    ``align_segments(["Tong", "tien:74,0OO"], "Tổng tiền: 74,000")`` -> ``["Tổng", "tiền: 74,000"]``.
    """
    if not parts:
        return []
    if len(parts) == 1:
        return [gold.strip()]
    hyp = " ".join(parts)
    pos = _alignment_map(hyp, gold)
    cuts: List[int] = []
    offset = 0
    for p in parts[:-1]:
        offset += len(p)
        cuts.append(_snap_to_space(gold, pos[offset]))
        offset += 1                                   # the joining space
    bounds = [0] + cuts + [len(gold)]
    for k in range(1, len(bounds)):                   # keep the cuts in order
        bounds[k] = max(bounds[k], bounds[k - 1])
    return [gold[bounds[k]:bounds[k + 1]].strip() for k in range(len(parts))]


# ---------------------------------------------------------------------------------- pairs of one receipt
def receipt_pairs(meta: Dict[str, Any], instances: Sequence[Any], config: Optional[RegionPairConfig] = None
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``line`` and ``region`` pairs of one receipt (boxes in processed-image pixels) and skip counters.

    ``meta`` is the OCR JSON of the receipt, ``instances`` its ``AnnotationInstance`` list.
    """
    cfg = config or RegionPairConfig()
    stats = {"lines": 0, "regions": 0, "line_skipped_cer": 0, "region_skipped_cer": 0, "region_skipped_empty": 0}
    geom = ImageGeometry.from_ocr_meta(meta)
    if any(frame_status((i.width, i.height), (geom.orig_w, geom.orig_h)) != "ok" for i in instances):
        stats["frame_mismatch"] = 1
        return [], stats
    items = [it for it in (meta.get("ocr_items") or []) if (it.get("text") or "").strip()]
    pairs: List[Dict[str, Any]] = []
    for region in map_annotation_regions(instances, geom):
        gold = unicodedata.normalize("NFC", region.text or "").strip()
        if not gold:
            continue
        pairs.append({"kind": "line", "label": region.label, "text": gold, "bbox": [float(v) for v in region.bbox]})
        stats["lines"] += 1

        inside = [it for it in items if contains_ratio(it["bbox"], region.bbox) >= cfg.min_containment]
        inside.sort(key=lambda it: it["bbox"][0])
        if not inside:
            continue
        texts = [unicodedata.normalize("NFC", it["text"]).strip() for it in inside]
        if cer(normalize_text(gold), normalize_text(" ".join(texts))) > cfg.max_line_cer:
            stats["line_skipped_cer"] += 1
            continue
        for it, seg, hyp in zip(inside, align_segments(texts, gold), texts):
            if not seg:
                stats["region_skipped_empty"] += 1
                continue
            if cer(normalize_text(seg), normalize_text(hyp)) > cfg.max_segment_cer:
                stats["region_skipped_cer"] += 1
                continue
            pairs.append({"kind": "region", "label": region.label, "text": seg,
                          "bbox": [float(v) for v in it["bbox"]], "ocr": hyp})
            stats["regions"] += 1
    return pairs, stats


def in_vocab(text: str, vocab_chars: str) -> bool:
    """True when VietOCR can encode every character of ``text`` (spaces are part of its vocabulary)."""
    allowed = set(vocab_chars)
    return all(c in allowed for c in text)


def crop_with_margin(image_bgr: Any, bbox: Sequence[float], margin: int = CROP_MARGIN
                     ) -> Tuple[Any, List[int]]:
    """Crop ``bbox`` plus ``margin`` pixels (clipped) and return it with the box position inside the crop (xyxy)."""
    import numpy as np
    h, w = image_bgr.shape[:2]
    x0, y0 = int(np.floor(bbox[0])), int(np.floor(bbox[1]))
    x1, y1 = int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(w, x1 + margin), min(h, y1 + margin)
    crop = np.ascontiguousarray(image_bgr[cy0:cy1, cx0:cx1])
    inner = [max(0, x0) - cx0, max(0, y0) - cy0, min(w, x1) - cx0, min(h, y1) - cy0]
    return crop, inner


def padded_view(crop: Any, inner: Sequence[int], pad_x: int, pad_y: int) -> Any:
    """The part of a saved crop that the recogniser would see with padding ``(pad_x, pad_y)`` around the box."""
    h, w = crop.shape[:2]
    x0, y0 = max(0, inner[0] - pad_x), max(0, inner[1] - pad_y)
    x1, y1 = min(w, inner[2] + pad_x), min(h, inner[3] + pad_y)
    return crop[y0:y1, x0:x1]


# ---------------------------------------------------------------------------------- whole dataset
def build_pair_set(csv_path: str, ocr_dir: str, image_dir: str, split_manifest: str, splits: Sequence[str],
                   out_dir: str, vocab_chars: Optional[str] = None, config: Optional[RegionPairConfig] = None
                   ) -> Dict[str, Any]:
    """Write ``<out_dir>/<split>/crops/*.png`` and ``<out_dir>/<split>/pairs.jsonl``; return counts per split.

    ``image_dir`` holds the processed images of the OCR run (``<file_id>.jpg``); ``ocr_dir`` its JSON files.
    Pairs with characters outside ``vocab_chars`` (when given) are counted and skipped.
    """
    import cv2

    records, _ = load_and_validate_mcocr_csv(csv_path)
    by_id = {r.img_id: r for r in records if r.is_valid and r.instances}
    split_of = read_split_manifest(split_manifest)
    report: Dict[str, Any] = {}
    for split in splits:
        crop_dir = os.path.join(out_dir, split, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        counts = {"receipts": 0, "missing_ocr": 0, "missing_image": 0, "frame_mismatch": 0, "out_of_vocab": 0,
                  "too_small": 0, "line": 0, "region": 0, "line_skipped_cer": 0, "region_skipped_cer": 0,
                  "region_skipped_empty": 0}
        with open(os.path.join(out_dir, split, "pairs.jsonl"), "w", encoding="utf-8") as out:
            for img_id in sorted(i for i, s in split_of.items() if s == split and i in by_id):
                file_id = os.path.splitext(img_id)[0]
                json_path = os.path.join(ocr_dir, f"{file_id}.json")
                if not os.path.exists(json_path):
                    counts["missing_ocr"] += 1
                    continue
                image = cv2.imread(os.path.join(image_dir, f"{file_id}.jpg"))
                if image is None:
                    counts["missing_image"] += 1
                    continue
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
                pairs, st = receipt_pairs(meta, by_id[img_id].instances, config)
                if st.get("frame_mismatch"):
                    counts["frame_mismatch"] += 1
                    continue
                counts["receipts"] += 1
                for k in ("line_skipped_cer", "region_skipped_cer", "region_skipped_empty"):
                    counts[k] += st[k]
                for n, p in enumerate(pairs):
                    if vocab_chars is not None and not in_vocab(p["text"], vocab_chars):
                        counts["out_of_vocab"] += 1
                        continue
                    crop, inner = crop_with_margin(image, p["bbox"])
                    if inner[2] - inner[0] < 4 or inner[3] - inner[1] < 4:
                        counts["too_small"] += 1
                        continue
                    name = f"{file_id}_{n:03d}_{p['kind']}.png"
                    cv2.imwrite(os.path.join(crop_dir, name), crop)
                    out.write(json.dumps({"crop": f"crops/{name}", "inner": inner, "text": p["text"],
                                          "kind": p["kind"], "label": p["label"], "img_id": img_id,
                                          "ocr": p.get("ocr")}, ensure_ascii=False) + "\n")
                    counts[p["kind"]] += 1
        report[split] = counts
    with open(os.path.join(out_dir, "pairs_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def load_pairs(split_dir: str) -> List[Dict[str, Any]]:
    """Rows of ``<split_dir>/pairs.jsonl`` with ``crop`` turned into an absolute path."""
    rows: List[Dict[str, Any]] = []
    with open(os.path.join(split_dir, "pairs.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["crop"] = os.path.join(split_dir, r["crop"])
                rows.append(r)
    return rows
