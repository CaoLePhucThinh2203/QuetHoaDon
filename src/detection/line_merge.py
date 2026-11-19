"""Merge neighbouring CRAFT boxes of one printed line into segments before recognition.

VietOCR (``vgg_transformer``) was trained to read whole text lines, but CRAFT often cuts one word or one number into
several boxes. Each crop is then read on its own and the character at the overlap is read twice ("399" + "9,000" for
"399,000", "210," + ",000"), or a short crop loses a digit. Reading the merged segment once avoids that.

Only boxes that touch or nearly touch are merged (gap <= ``max_gap`` x text height), so the columns of an item table
(name, unit price, amount) stay separate segments and the word positions given to LayoutXLM stay close to the truth
(``items_to_words`` splits a segment's text into word boxes by word length). A segment is never wider than
``max_aspect`` x its height: VietOCR resizes crops to 32 px high and at most 512 px wide, so a longer crop is squeezed.

Pure Python + numpy; off unless ``detection.merge_lines`` is true.
"""
from typing import List, Optional

import numpy as np

from src.detection.detector import TextRegion
from src.detection.postprocess import sort_reading_order


def _height(b: List[int]) -> float:
    return float(b[3] - b[1])


def _v_overlap(a: List[int], b: List[int]) -> float:
    """Vertical overlap as a share of the smaller height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, inter) / max(1.0, min(_height(a), _height(b)))


def _union(regions: List[TextRegion]) -> TextRegion:
    if len(regions) == 1:
        return regions[0]
    x0 = min(r.bbox[0] for r in regions)
    y0 = min(r.bbox[1] for r in regions)
    x1 = max(r.bbox[2] for r in regions)
    y1 = max(r.bbox[3] for r in regions)
    ox0 = min(r.orig_bbox[0] for r in regions)
    oy0 = min(r.orig_bbox[1] for r in regions)
    ox1 = max(r.orig_bbox[2] for r in regions)
    oy1 = max(r.orig_bbox[3] for r in regions)
    return TextRegion(polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], bbox=[x0, y0, x1, y1],
                      orig_bbox=[ox0, oy0, ox1, oy1], score=None)


def merge_segments(regions: List[TextRegion], max_gap: float = 1.0, min_v_overlap: float = 0.5,
                   max_aspect: float = 16.0, line_tolerance: Optional[float] = None) -> List[TextRegion]:
    """Merge boxes of the same line that touch or nearly touch; returns regions in reading order.

    * ``max_gap``: largest horizontal gap, in text heights (negative gaps, i.e. overlapping boxes, always qualify);
    * ``min_v_overlap``: the two boxes must overlap vertically by this share of the smaller height;
    * ``max_aspect``: a merged segment stays at most this many heights wide;
    * ``line_tolerance``: centre-y tolerance for one line (default 0.5 x median height, as ``sort_reading_order``).
    """
    if len(regions) < 2:
        return list(regions)
    med_h = float(np.median([_height(r.bbox) for r in regions]))
    tol = line_tolerance if line_tolerance is not None else max(4.0, 0.5 * med_h)

    def cy(r: TextRegion) -> float:
        return (r.bbox[1] + r.bbox[3]) / 2.0

    lines: List[dict] = []
    for r in sorted(regions, key=cy):
        if lines and abs(cy(r) - lines[-1]["cy"]) <= tol:
            lines[-1]["items"].append(r)
            lines[-1]["cy"] = float(np.mean([cy(x) for x in lines[-1]["items"]]))
        else:
            lines.append({"cy": cy(r), "items": [r]})

    merged: List[TextRegion] = []
    for ln in lines:
        segment: List[TextRegion] = []
        for r in sorted(ln["items"], key=lambda x: x.bbox[0]):
            if segment:
                last, seg = segment[-1], _union(segment)
                h = min(_height(last.bbox), _height(r.bbox))
                gap = r.bbox[0] - max(x.bbox[2] for x in segment)
                width = max(seg.bbox[2], r.bbox[2]) - seg.bbox[0]
                height = max(seg.bbox[3], r.bbox[3]) - min(seg.bbox[1], r.bbox[1])
                if (gap <= max_gap * h and _v_overlap(last.bbox, r.bbox) >= min_v_overlap
                        and width <= max_aspect * height):
                    segment.append(r)
                    continue
                merged.append(_union(segment))
            segment = [r]
        if segment:
            merged.append(_union(segment))
    return sort_reading_order(merged)
