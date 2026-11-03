"""Abstract interfaces for Vietnamese OCR text recognizers (VietOCR)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TextRecognitionResult:
    """Standardized output from a text recognition model."""
    text: str
    confidence: float


class BaseTextRecognizer(ABC):
    """Abstract base class for OCR text recognizers."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def recognize_crop(self, crop_image: Any) -> TextRecognitionResult:
        """Transcribe a cropped single-line image into Vietnamese text.

        Args:
            crop_image: Image patch (numpy array or PIL Image).

        Returns:
            TextRecognitionResult with transcribed text and confidence score.
        """
        pass

    def recognize_batch(self, crop_images: List[Any]) -> List[TextRecognitionResult]:
        """Transcribe a batch of cropped image lines."""
        return [self.recognize_crop(img) for img in crop_images]
