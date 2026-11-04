"""Text recognition: interface, OCR item structure and the VietOCR implementation.

``VietOCRRecognizer`` is imported explicitly from ``src.recognition.vietocr_recognizer``.
"""
from src.recognition.recognizer import BaseTextRecognizer, OcrItem, TextRecognitionResult

__all__ = ["BaseTextRecognizer", "OcrItem", "TextRecognitionResult"]
