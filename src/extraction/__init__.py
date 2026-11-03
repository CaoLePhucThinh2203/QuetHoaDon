"""Key information extraction schemas, label encodings, and entity definitions."""
from src.extraction.schema import (
    TargetEntity,
    BIO_LABELS,
    LABEL2ID,
    ID2LABEL,
    get_bio_label,
)

__all__ = [
    "TargetEntity",
    "BIO_LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "get_bio_label",
]
