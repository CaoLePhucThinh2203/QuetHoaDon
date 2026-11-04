"""Text detection: interface, post-processing and the CRAFT implementation.

``CraftDetector`` is not imported here on purpose: it needs torch and craft-text-detector.
Use ``from src.detection.craft_detector import CraftDetector``.
"""
from src.detection.detector import BaseTextDetector, TextRegion
from src.detection.postprocess import drop_nested, sort_reading_order

__all__ = ["BaseTextDetector", "TextRegion", "drop_nested", "sort_reading_order"]
