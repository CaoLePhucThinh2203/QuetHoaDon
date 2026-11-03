"""Common utility modules for geometry, I/O, and data representations."""
from src.common.bbox_utils import (
    normalize_bbox,
    denormalize_bbox,
    polygon_to_bbox,
    compute_iou,
)

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "polygon_to_bbox",
    "compute_iou",
]
