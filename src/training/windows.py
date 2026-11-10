"""Sliding windows for receipts longer than the 512-token limit of LayoutXLM (pure Python / NumPy).

A receipt is a list of words; each word becomes one or more sub-tokens. ``make_windows`` cuts the word list into
overlapping windows whose sub-token count stays below a budget, and ``merge_window_probs`` averages the
predictions that several windows give for the same word.
"""
from typing import List, Sequence, Tuple

import numpy as np


def make_windows(token_counts: Sequence[int], max_tokens: int = 500, overlap_tokens: int = 100) -> List[Tuple[int, int]]:
    """Cut ``len(token_counts)`` words into ``(start, end)`` windows (``end`` exclusive).

    Every window holds consecutive words whose sub-token counts add up to at most ``max_tokens`` (a single word
    larger than the budget gets a window of its own and is truncated later by the tokenizer). Consecutive windows
    share roughly ``overlap_tokens`` tokens so that a word near a cut sees context on both sides. Every word is in
    at least one window and the loop always advances.
    """
    n = len(token_counts)
    windows: List[Tuple[int, int]] = []
    if n == 0:
        return windows
    start = 0
    while True:
        end, total = start, 0
        while end < n and (end == start or total + token_counts[end] <= max_tokens):
            total += token_counts[end]
            end += 1
        windows.append((start, end))
        if end >= n:
            break
        new_start, back = end, 0
        while new_start > start + 1 and back + token_counts[new_start - 1] <= overlap_tokens:
            new_start -= 1
            back += token_counts[new_start]
        start = new_start
    return windows


def merge_window_probs(n_words: int, n_labels: int,
                       window_results: Sequence[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Average per-word probabilities over all windows that cover the word.

    ``window_results`` is a list of ``(start, probs)`` with ``probs`` of shape ``[k, n_labels]`` for words
    ``start .. start + k - 1`` (``k`` may be smaller than the window when the tokenizer truncated it).
    Returns ``(probs [n_words, n_labels], covered [n_words])``; words no window predicted get probability 1 for
    label 0 (``O``).
    """
    acc = np.zeros((n_words, n_labels), dtype=np.float64)
    cnt = np.zeros(n_words, dtype=np.float64)
    for start, probs in window_results:
        k = min(len(probs), n_words - start)
        if k <= 0:
            continue
        acc[start:start + k] += probs[:k]
        cnt[start:start + k] += 1
    covered = cnt > 0
    out = np.zeros_like(acc)
    out[covered] = acc[covered] / cnt[covered][:, None]
    out[~covered, 0] = 1.0
    return out, covered
