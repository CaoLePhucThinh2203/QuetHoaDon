"""Recognizer interface, the OCR item data structure and the region-cropping logic."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.detection.detector import TextRegion


@dataclass
class TextRecognitionResult:
    text: str
    confidence: Optional[float] = None   # None when the engine exposes no calibrated confidence


@dataclass
class OcrItem:
    """A detected region plus its transcription (the unit stored in the OCR JSON)."""
    region_id: int
    reading_order: int
    polygon: List[List[int]]
    bbox: List[int]                      # xyxy, processed-image coordinates
    orig_bbox: List[int]
    text: str
    ocr_confidence: Optional[float] = None
    detection_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseTextRecognizer(ABC):
    """Abstract OCR engine. Subclasses implement ``recognize_batch`` on RGB numpy crops."""

    def __init__(self, config: Optional[dict] = None, batch_size: int = 32) -> None:
        self.config = config or {}
        self.batch_size = batch_size
        self.n_errors = 0

    @abstractmethod
    def recognize_batch(self, crops_rgb: List[np.ndarray]) -> List[TextRecognitionResult]:
        """Transcribe single-line crops; must return one result per crop, in order."""

    def recognize_crop(self, crop_rgb: np.ndarray) -> TextRecognitionResult:
        return self.recognize_batch([crop_rgb])[0]

    def recognize_regions(self, image_bgr: np.ndarray, regions: List[TextRegion],
                          pad_x: int = 3, pad_y: int = 2) -> List[OcrItem]:
        """Crop every region (small padding), transcribe in batches, drop empty transcriptions."""
        h_img, w_img = image_bgr.shape[:2]
        rgb = image_bgr[:, :, ::-1]
        crops, kept = [], []
        for r in regions:
            x0, y0, x1, y1 = r.bbox
            x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
            x1, y1 = min(w_img, x1 + pad_x), min(h_img, y1 + pad_y)
            if (x1 - x0) < 4 or (y1 - y0) < 4:
                continue
            crops.append(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            kept.append(r)

        items: List[OcrItem] = []
        for i in range(0, len(crops), self.batch_size):
            results = self.recognize_batch(crops[i:i + self.batch_size])
            for r, res in zip(kept[i:i + self.batch_size], results):
                text = (res.text or "").strip()
                if text:
                    items.append(OcrItem(
                        region_id=len(items), reading_order=len(items),
                        polygon=r.polygon, bbox=r.bbox, orig_bbox=r.orig_bbox, text=text,
                        ocr_confidence=res.confidence, detection_score=r.score))
        return items
