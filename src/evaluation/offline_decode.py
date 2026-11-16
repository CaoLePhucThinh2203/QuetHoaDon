"""Decode saved word probabilities into the final fields without a GPU (pure Python / NumPy).

Uses exactly the decoding that scored the Week-4 results (soft decoding of the total and the date, then the keyword /
date rules for empty fields), so changes to the decoding or to the rules can be tried and scored in seconds on the
saved probabilities of a checkpoint.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.extraction.rules import fill_missing
from src.training.field_eval import decode_predictions_soft

Preds = Dict[str, Dict[str, Optional[str]]]


def decode_from_probs(records: Sequence[Dict[str, Any]], probs_by_id: Dict[str, np.ndarray]
                      ) -> Tuple[Preds, List[Dict[str, Any]]]:
    """Final predictions and the ``merged`` structure (``probs``, ``covered``) of every record, in record order."""
    merged: List[Dict[str, Any]] = []
    for rec in records:
        img_id = rec["img_id"]
        if img_id not in probs_by_id:
            raise KeyError(f"no saved probabilities for {img_id}")
        probs = np.asarray(probs_by_id[img_id])
        if probs.ndim != 2 or len(probs) != len(rec["words"]):
            raise ValueError(f"{img_id}: probabilities {probs.shape} do not match {len(rec['words'])} words")
        merged.append({"probs": probs, "covered": np.ones(len(probs), dtype=bool)})
    soft = decode_predictions_soft(records, merged)
    return fill_missing(records, soft), merged
