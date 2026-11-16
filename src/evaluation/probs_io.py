"""Save and load per-word label probabilities so decoding experiments need no GPU (NumPy only)."""
from typing import Dict, Sequence

import numpy as np


def save_probs(path: str, ids: Sequence[str], probs: Sequence[np.ndarray]) -> None:
    """One compressed file: ``ids`` plus ``p0, p1, ...`` (arrays ``[n_words, n_labels]`` stored as given, same order).

    The evaluation produces float64; keeping that precision makes offline decoding reproduce the reported predictions exactly.
    """
    if len(ids) != len(probs):
        raise ValueError("ids and probs must have the same length")
    arrays = {f"p{i}": np.asarray(p) for i, p in enumerate(probs)}
    np.savez_compressed(path, ids=np.array(list(ids), dtype=str), **arrays)


def load_probs(path: str) -> Dict[str, np.ndarray]:
    """``{img_id: probs}`` written by ``save_probs``."""
    with np.load(path, allow_pickle=False) as z:
        ids = [str(i) for i in z["ids"]]
        return {img_id: z[f"p{i}"] for i, img_id in enumerate(ids)}
