"""Geometry and bounding box manipulation utilities for OCR and LayoutXLM.

Box conventions used across the repository
------------------------------------------
* ``xyxy``  = [x0, y0, x1, y1]  -> canonical *internal* format (pipeline, IoU, LayoutXLM).
* ``xywh``  = [x, y, w, h]      -> format of the raw MC-OCR annotations (``anno_polygons.bbox``).

Always convert MC-OCR boxes with :func:`xywh_to_xyxy` before comparing them with
detector output; mixing the two formats silently produces wrong IoU values.
"""
from typing import List, Tuple, Union


def polygon_to_bbox(segmentation: List[Union[float, int, List]]) -> List[int]:
    """Convert a flat or nested polygon coordinate list [x1, y1, x2, y2, ...] to an axis-aligned
    bounding box [x_min, y_min, width, height].
    """
    if not segmentation:
        raise ValueError("Segmentation list cannot be empty.")

    # Handle nested list if wrapped in [[x1, y1, ...]] or list of points
    if isinstance(segmentation[0], list):
        flat_coords = [c for pt in segmentation for c in (pt if isinstance(pt, list) else [pt])]
    else:
        flat_coords = segmentation

    if len(flat_coords) < 4:
        raise ValueError("Segmentation list must contain at least 4 coordinates (2 points).")

    xs = flat_coords[0::2]
    ys = flat_coords[1::2]

    min_x = int(min(xs))
    max_x = int(max(xs))
    min_y = int(min(ys))
    max_y = int(max(ys))

    w = max(0, max_x - min_x)
    h = max(0, max_y - min_y)

    return [min_x, min_y, w, h]


def normalize_bbox(
    bbox: List[Union[int, float]],
    width: int,
    height: int,
    scale: int = 1000,
    is_xywh: bool = True,
) -> List[int]:
    """Normalize bounding box coordinates to integer range [0, scale] for LayoutXLM.

    Args:
        bbox: Bounding box coordinates. Either [x, y, w, h] or [x0, y0, x1, y1].
        width: Original image width in pixels.
        height: Original image height in pixels.
        scale: Normalization upper bound (default: 1000 for LayoutLM/LayoutXLM).
        is_xywh: True if bbox format is [x, y, w, h], False if [x0, y0, x1, y1].

    Returns:
        [x0_norm, y0_norm, x1_norm, y1_norm] with 0 <= x0 <= x1 <= scale and 0 <= y0 <= y1 <= scale.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive integers, got ({width}, {height})")

    if is_xywh:
        x, y, w, h = bbox
        x0 = float(x)
        y0 = float(y)
        x1 = float(x + w)
        y1 = float(y + h)
    else:
        x0, y0, x1, y1 = [float(v) for v in bbox]

    # Ensure canonical ordering
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)

    norm_x0 = max(0, min(scale, int(round((x_min / width) * scale))))
    norm_y0 = max(0, min(scale, int(round((y_min / height) * scale))))
    norm_x1 = max(0, min(scale, int(round((x_max / width) * scale))))
    norm_y1 = max(0, min(scale, int(round((y_max / height) * scale))))

    # Guarantee monotonic invariant: x0 <= x1, y0 <= y1
    return [
        min(norm_x0, norm_x1),
        min(norm_y0, norm_y1),
        max(norm_x0, norm_x1),
        max(norm_y0, norm_y1),
    ]


def denormalize_bbox(
    norm_bbox: List[int],
    width: int,
    height: int,
    scale: int = 1000,
) -> List[int]:
    """Convert normalized [x0, y0, x1, y1] coordinates back to original image pixel space."""
    x0, y0, x1, y1 = norm_bbox
    orig_x0 = int(round((x0 / scale) * width))
    orig_y0 = int(round((y0 / scale) * height))
    orig_x1 = int(round((x1 / scale) * width))
    orig_y1 = int(round((y1 / scale) * height))

    return [orig_x0, orig_y0, orig_x1, orig_y1]


def compute_iou(boxA: List[Union[int, float]], boxB: List[Union[int, float]]) -> float:
    """Compute Intersection over Union (IoU) of two [x0, y0, x1, y1] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    boxB_area = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def xywh_to_xyxy(box: List[Union[int, float]]) -> List[float]:
    """[x, y, w, h] -> [x0, y0, x1, y1]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: List[Union[int, float]]) -> List[float]:
    """[x0, y0, x1, y1] -> [x, y, w, h]."""
    x0, y0, x1, y1 = box
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_to_xyxy(segmentation: List[Union[float, int, List]]) -> List[int]:
    """Axis-aligned [x0, y0, x1, y1] box around a polygon (flat or nested)."""
    x, y, w, h = polygon_to_bbox(segmentation)
    return [x, y, x + w, y + h]


def contains_ratio(inner: List[Union[int, float]], outer: List[Union[int, float]]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (both xyxy)."""
    iw = min(inner[2], outer[2]) - max(inner[0], outer[0])
    ih = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = max(0.0, float(inner[2] - inner[0])) * max(0.0, float(inner[3] - inner[1]))
    if iw <= 0 or ih <= 0 or area <= 0:
        return 0.0
    return float(iw * ih) / area
