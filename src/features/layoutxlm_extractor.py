"""LayoutXLM feature extraction (pretrained encoder, no task head)."""
from typing import Any, Dict, List, Optional, Tuple

from src.common.bbox_utils import normalize_bbox
from src.recognition.recognizer import OcrItem


def split_box_by_tokens(box: List[int], tokens: List[str]) -> List[List[int]]:
    """Split an xyxy box horizontally between words, proportionally to word length (+ spaces)."""
    total_w = max(1, box[2] - box[0])
    total_chars = sum(len(t) for t in tokens) + (len(tokens) - 1)
    out, cursor = [], float(box[0])
    for t in tokens:
        w = total_w * len(t) / total_chars
        out.append([int(cursor), box[1], int(round(cursor + w)), box[3]])
        cursor += w + total_w / total_chars
    return out


def build_words_and_boxes(items: List[OcrItem], width: int, height: int,
                          max_words: Optional[int] = None) -> Tuple[List[str], List[List[int]]]:
    """Words and LayoutXLM-normalised ([0, 1000]) boxes from OCR items, in reading order."""
    words: List[str] = []
    boxes: List[List[int]] = []
    for item in items:
        tokens = item.text.split()
        if not tokens:
            continue
        for tok, b in zip(tokens, split_box_by_tokens(item.bbox, tokens)):
            words.append(tok)
            boxes.append(normalize_bbox(b, width, height, scale=1000, is_xywh=False))
    if max_words is not None:
        words, boxes = words[:max_words], boxes[:max_words]
    return words, boxes


class LayoutXLMFeatureExtractor:
    def __init__(self, model_name: str = "microsoft/layoutxlm-base", device: str = "cuda",
                 max_tokens: int = 512, save_fp16: bool = True) -> None:
        import torch  # noqa: F401  (fail early if missing)
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.max_tokens = max_tokens
        self.save_fp16 = save_fp16
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def extract(self, image_pil: Any, items: List[OcrItem]) -> Dict[str, Any]:
        import torch
        w, h = image_pil.size
        words, boxes = build_words_and_boxes(items, w, h, max_words=self.max_tokens - 2)
        if not words:
            return {"last_hidden_state": None}

        enc = self.processor(images=image_pil, text=words, boxes=boxes, return_tensors="pt",
                             truncation=True, padding="max_length", max_length=self.max_tokens)
        inputs = {k: enc[k].to(self.device) for k in ("input_ids", "bbox", "attention_mask")}
        image_tensor = enc.get("image", enc.get("pixel_values"))
        if image_tensor is not None:
            inputs["image"] = image_tensor.to(self.device)

        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=False)

        hs = out.last_hidden_state.cpu()
        if self.save_fp16:
            hs = hs.half()
        n_tokens = int(enc["attention_mask"].sum())
        return {
            "last_hidden_state": hs,
            "input_ids": enc["input_ids"],
            "normalized_bboxes": enc["bbox"],
            "attention_mask": enc["attention_mask"],
            "words": words,
            "possibly_truncated": n_tokens >= self.max_tokens,
        }
