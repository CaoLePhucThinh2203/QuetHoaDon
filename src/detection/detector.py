"""Detector interface and the standard text-region data structure."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TextRegion:
    """One detected text region. All boxes are ``xyxy``."""
    polygon: List[List[int]]          # [[x, y], ...] in processed-image coordinates
    bbox: List[int]                   # [x0, y0, x1, y1] in processed-image coordinates
    orig_bbox: List[int]              # bbox mapped back by the resize factor (exact only if skew_angle == 0)
    score: Optional[float] = None     # detection confidence; None if the detector does not expose one

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseTextDetector(ABC):
    """Abstract text detector (CRAFT, DBNet, ...)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def detect(self, image_bgr: Any, scale_ratio: float = 1.0) -> List[TextRegion]:
        """Return regions in reading order."""
