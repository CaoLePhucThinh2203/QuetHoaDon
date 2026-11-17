"""Pure geometric post-processing of detected regions (no model dependencies)."""
from typing import List

import numpy as np

from src.common.bbox_utils import contains_ratio
from src.detection.detector import TextRegion


def _area(r: TextRegion) -> int:
    return max(1, (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))


def drop_nested(regions: List[TextRegion], thr: float = 0.8) -> List[TextRegion]:
    """Remove a region when >= ``thr`` of its area lies inside a larger one.

    Fixes CRAFT's habit of returning a separate tiny box for the comma/dot in '27,000',
    which VietOCR would otherwise read as garbage. Identical duplicates keep one copy.
    """
    areas = [_area(r) for r in regions]
    keep: List[TextRegion] = []
    for i, r in enumerate(regions):
        nested = False
        for j, o in enumerate(regions):
            if i == j or areas[j] < areas[i] or (areas[j] == areas[i] and j > i):
                continue
            if contains_ratio(r.bbox, o.bbox) >= thr:
                nested = True
                break
        if not nested:
            keep.append(r)
    return keep


def sort_reading_order(regions: List[TextRegion]) -> List[TextRegion]:
    """Group regions into lines by centre-y (tolerance = 0.5 x median height), then left -> right."""
    if not regions:
        return regions
    med_h = float(np.median([r.bbox[3] - r.bbox[1] for r in regions]))
    thr = max(4.0, 0.5 * med_h)

    def cy(r: TextRegion) -> float:
        return (r.bbox[1] + r.bbox[3]) / 2.0

    lines: List[dict] = []
    for r in sorted(regions, key=cy):
        if lines and abs(cy(r) - lines[-1]["cy"]) <= thr:
            ln = lines[-1]
            ln["items"].append(r)
            ln["cy"] = float(np.mean([cy(x) for x in ln["items"]]))
        else:
            lines.append({"cy": cy(r), "items": [r]})

    ordered: List[TextRegion] = []
    for ln in lines:
        ordered.extend(sorted(ln["items"], key=lambda r: r.bbox[0]))
    return ordered


def filter_nested_items(items: List[dict], thr: float) -> List[dict]:
    """``drop_nested`` applied to the ``ocr_items`` of an OCR JSON (order kept, ``region_id`` renumbered).

    Lets a stricter nested threshold be tried on an existing OCR run without detecting and reading again. It is the
    same as re-running the pipeline with ``nested_threshold=thr``, except that boxes whose reading was empty (and were
    therefore not saved) can no longer remove the boxes inside them.
    """
    regions = [TextRegion(polygon=it.get("polygon") or [], bbox=list(it["bbox"]), orig_bbox=list(it.get("orig_bbox") or it["bbox"]))
               for it in items]
    kept_ids = {id(r) for r in drop_nested(regions, thr)}
    out: List[dict] = []
    for it, r in zip(items, regions):
        if id(r) in kept_ids:
            new = dict(it)
            new["region_id"] = new["reading_order"] = len(out)
            out.append(new)
    return out
