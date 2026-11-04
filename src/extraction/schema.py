"""Canonical output fields for receipt information extraction.

These are semantic output keys, not BIO training labels (the BIO tag set of the Week-4
token classifier is derived from ``ANNOTATED_FIELDS``).
"""
from enum import Enum
from typing import Dict, List


class TargetEntity(str, Enum):
    SELLER = "SELLER"
    ADDRESS = "ADDRESS"
    ITEM_NAME = "ITEM_NAME"
    ITEM_PRICE = "ITEM_PRICE"
    TOTAL_COST = "TOTAL_COST"
    QUANTITY = "QUANTITY"
    TIMESTAMP = "TIMESTAMP"


REQUIRED_OUTPUT_FIELDS: List[str] = [
    TargetEntity.SELLER.value,
    TargetEntity.ADDRESS.value,
    TargetEntity.ITEM_NAME.value,
    TargetEntity.ITEM_PRICE.value,
    TargetEntity.TOTAL_COST.value,
]

OPTIONAL_OUTPUT_FIELDS: List[str] = [
    TargetEntity.QUANTITY.value,
    TargetEntity.TIMESTAMP.value,
]


# The only fields MC-OCR annotates, hence the only ones that can be scored quantitatively.
ANNOTATED_FIELDS: List[str] = [
    TargetEntity.SELLER.value,
    TargetEntity.ADDRESS.value,
    TargetEntity.TIMESTAMP.value,
    TargetEntity.TOTAL_COST.value,
]


def empty_field() -> Dict[str, object]:
    """Create an explicit empty extraction result; never invent a value."""
    return {
        "text": None,
        "bbox": None,
        "confidence": None,
        "status": "not_found",
    }


def empty_receipt_result() -> Dict[str, object]:
    """Return the canonical receipt result structure."""
    return {
        "seller": empty_field(),
        "address": empty_field(),
        "timestamp": empty_field(),
        "items": [],
        "total": empty_field(),
    }
