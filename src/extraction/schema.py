"""Entity classification schema and BIO tag encodings for LayoutXLM."""
from enum import Enum
from typing import Dict, List, Optional


class TargetEntity(str, Enum):
    """Core key information fields extracted from Vietnamese receipts."""
    SELLER = "SELLER"
    ADDRESS = "ADDRESS"
    TIMESTAMP = "TIMESTAMP"
    TOTAL_COST = "TOTAL_COST"


# 9-class BIO Sequence Labeling Scheme
BIO_LABELS: List[str] = [
    "O",
    "B-SELLER",
    "I-SELLER",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TIMESTAMP",
    "I-TIMESTAMP",
    "B-TOTAL_COST",
    "I-TOTAL_COST",
]

LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(BIO_LABELS)}


def get_bio_label(entity: Optional[str], is_first_token: bool = True) -> str:
    """Return BIO tag string given entity name and token position."""
    if not entity or entity.upper() == "O" or entity.upper() == "OTHER":
        return "O"

    clean_entity = entity.upper()
    prefix = "B-" if is_first_token else "I-"
    full_label = f"{prefix}{clean_entity}"

    if full_label not in LABEL2ID:
        return "O"

    return full_label
