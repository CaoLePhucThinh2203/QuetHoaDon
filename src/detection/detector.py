"""Abstract interfaces and geometric bounding box postprocessing for CRAFT text detection."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


@dataclass
class TextDetectionResult:
    """Standardized output from a text detection model."""
    polygons: List[List[List[float]]]  # List of 4-point polygons [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
    boxes: List[List[int]]             # Axis-aligned bounding boxes [[x, y, w, h], ...]
    confidences: List[float]           # Detection confidence scores


class BaseTextDetector(ABC):
    """Abstract base class for document text detectors (CRAFT, DBNet, etc.)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def detect(self, image: Any) -> TextDetectionResult:
        """Run text detection on an input image.

        Args:
            image: Image object (numpy array or PIL Image).

        Returns:
            TextDetectionResult with detected polygons, bounding boxes, and confidences.
        """
        pass

    @staticmethod
    def polygon_to_xywh(polygon: List[List[float]]) -> List[int]:
        """Convert a 4-point polygon to [x, y, w, h]."""
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        x_min = int(min(xs))
        y_min = int(min(ys))
        w = max(0, int(max(xs)) - x_min)
        h = max(0, int(max(ys)) - y_min)
        return [x_min, y_min, w, h]
