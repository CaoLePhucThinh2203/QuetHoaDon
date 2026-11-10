"""Datasets for LayoutXLM token classification (needs torch, Pillow and transformers; runs on Colab).

One dataset item is one *window* (a slice of a receipt's words, see ``windows.make_windows``) so receipts longer
than 512 sub-tokens are split instead of truncated. The processor is called in the batched form
(``images=[image], text=[words]``) that was verified on Colab with transformers 5.x, then the batch dimension is
dropped.
"""
import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.token_dataset import LABEL2ID
from src.training.windows import make_windows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_token_counts(tokenizer: Any, words: Sequence[str]) -> List[int]:
    """Sub-token count of every word (>= 1). Falls back to a length estimate if the tokenizer cannot tokenize."""
    counts: List[int] = []
    for w in words:
        try:
            counts.append(max(1, len(tokenizer.tokenize(w))))
        except Exception:                                   # pragma: no cover - defensive
            counts.append(max(1, math.ceil(len(w) / 3)))
    return counts


def build_windows(records: Sequence[Dict[str, Any]], tokenizer: Any, max_tokens: int = 480,
                  overlap_tokens: int = 100) -> List[Tuple[int, int, int]]:
    """``(doc_index, start_word, end_word)`` for every window of every receipt (empty receipts give none)."""
    out: List[Tuple[int, int, int]] = []
    for d, rec in enumerate(records):
        counts = word_token_counts(tokenizer, rec["words"])
        for s, e in make_windows(counts, max_tokens, overlap_tokens):
            out.append((d, s, e))
    return out


class WindowDataset(Dataset):
    """Items: ``input_ids``, ``attention_mask``, ``bbox``, ``labels``, ``image`` plus ``doc`` and ``start``.

    ``labels`` hold the gold label id on the first sub-token of each word and -100 elsewhere; the model never
    receives them (the loss is computed in the engine), they only mark which positions belong to words.
    """

    def __init__(self, records: Sequence[Dict[str, Any]], windows: Sequence[Tuple[int, int, int]], processor: Any,
                 image_dir: str, max_length: int = 512) -> None:
        self.records = records
        self.windows = list(windows)
        self.processor = processor
        self.image_dir = image_dir
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        d, s, e = self.windows[i]
        rec = self.records[d]
        with Image.open(os.path.join(self.image_dir, rec["image"])) as im:
            image = im.convert("RGB")
        enc = self.processor(
            images=[image], text=[rec["words"][s:e]], boxes=[rec["boxes"][s:e]],
            word_labels=[[LABEL2ID[t] for t in rec["labels"][s:e]]],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v[0] for k, v in enc.items()}
        if "pixel_values" in item and "image" not in item:
            item["image"] = item.pop("pixel_values")
        item["doc"] = d
        item["start"] = s
        return item
