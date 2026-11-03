"""Canonical fields and output structure for unsupervised receipt extraction."""
from src.extraction.schema import (
    TargetEntity,
    REQUIRED_OUTPUT_FIELDS,
    OPTIONAL_OUTPUT_FIELDS,
    empty_field,
    empty_receipt_result,
)

__all__ = [
    "TargetEntity",
    "REQUIRED_OUTPUT_FIELDS",
    "OPTIONAL_OUTPUT_FIELDS",
    "empty_field",
    "empty_receipt_result",
]
