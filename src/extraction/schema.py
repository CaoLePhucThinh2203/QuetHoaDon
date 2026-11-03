"""Canonical output fields for unsupervised receipt information extraction.

These are semantic output keys, not BIO training labels. The project does not
train a supervised token-classification head from receipt entity annotations.
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
        "items": [],
        "total": empty_field(),
    }
