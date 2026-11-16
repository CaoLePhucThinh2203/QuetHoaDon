"""End-to-end receipt extraction: image -> OCR -> LayoutXLM token classifier -> fields (Week 5 demo).

Data -> Model -> Output
* data: a receipt photo (or an OCR result that already exists);
* model: the OCR pipeline (preprocessing, CRAFT, VietOCR) followed by the fine-tuned LayoutXLM token classifier;
* output: ``SELLER``, ``ADDRESS``, ``TIMESTAMP`` and ``TOTAL_COST`` plus per-stage timing.

The fields are decoded with **exactly the functions that scored the Week-4 results** (soft decoding of the total and
the date, then the keyword / date rules for empty fields), so a demo result is produced the same way as a scored one.

The class is split so that everything except the neural network is plain Python and unit-tested: the network is a
function ``probs_fn(record, image_dir) -> array [n_words, 9]`` that ``from_checkpoint`` builds with torch.
"""
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.common.bbox_utils import normalize_bbox
from src.evaluation.metrics import normalize_amount
from src.extraction.rules import fill_missing
from src.extraction.schema import ANNOTATED_FIELDS
from src.preprocessing.token_dataset import ID2LABEL, items_to_words
from src.training.field_eval import decode_predictions_soft, normalize_date

ProbsFn = Callable[[Dict[str, Any], str], np.ndarray]


# ---------------------------------------------------------------------------------- pure Python
def record_from_ocr(meta: Dict[str, Any], image_name: str) -> Dict[str, Any]:
    """Token record (words, boxes normalised to [0, 1000], placeholder ``O`` labels) from an OCR result.

    Uses the same word splitting and box normalisation as ``build_token_dataset`` (training) and the LayoutXLM
    feature extractor, so inference sees the words the model was trained on. ``image_name`` is the preprocessed
    image inside the directory handed to ``probs_fn``.
    """
    words = items_to_words(meta.get("ocr_items") or [])
    height, width = meta["processed_shape"][:2]
    return {
        "img_id": meta.get("file_id") or os.path.splitext(image_name)[0],
        "image": image_name,
        "words": [w["text"] for w in words],
        "boxes": [normalize_bbox(w["bbox"], width, height, scale=1000, is_xywh=False) for w in words],
        "labels": ["O"] * len(words),
        "gt": {},
        "pixel_boxes": [list(w["bbox"]) for w in words],
    }


def fields_from_probs(record: Dict[str, Any], probs: np.ndarray) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Decode word probabilities into the four fields and the predicted BIO tag of every word.

    Each field has ``text`` (or ``None``), ``status`` (``found`` or ``not_found``) and ``source`` (``model`` when the
    value comes from the tagger, ``rule`` when the keyword / date rule supplied or replaced it). ``TOTAL_COST`` also
    carries ``amount_vnd`` and ``TIMESTAMP`` carries ``date`` (``dd/mm/yyyy``) when they can be read.
    """
    img_id = record["img_id"]
    merged = [{"probs": probs, "covered": np.ones(len(probs), dtype=bool)}]
    soft = decode_predictions_soft([record], merged)
    final = fill_missing([record], soft)
    fields: Dict[str, Dict[str, Any]] = {}
    for name in ANNOTATED_FIELDS:
        text = final[img_id].get(name)
        source = None if not text else ("rule" if text != soft[img_id].get(name) else "model")
        fields[name] = {"text": text, "status": "found" if text else "not_found", "source": source}
    total = fields["TOTAL_COST"]["text"]
    fields["TOTAL_COST"]["amount_vnd"] = normalize_amount(total) if total else None
    fields["TIMESTAMP"]["date"] = normalize_date(fields["TIMESTAMP"]["text"])
    tags = [ID2LABEL[int(i)] for i in probs.argmax(axis=1)] if len(probs) else []
    return fields, tags


def empty_fields() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {n: {"text": None, "status": "not_found", "source": None} for n in ANNOTATED_FIELDS}
    out["TOTAL_COST"]["amount_vnd"] = None
    out["TIMESTAMP"]["date"] = None
    return out


# ---------------------------------------------------------------------------------- the extractor
class ReceiptExtractor:
    """OCR pipeline + token classifier. ``ocr_pipeline`` may be ``None`` when only ``extract_from_ocr`` is used."""

    def __init__(self, probs_fn: ProbsFn, ocr_pipeline: Any = None) -> None:
        self.probs_fn = probs_fn
        self.ocr_pipeline = ocr_pipeline

    def _run(self, record: Dict[str, Any], image_dir: str, boxes: List[List[float]], bbox_unit: str,
             timing_in: Optional[Dict[str, float]], skew_angle: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if record["words"]:
            probs = np.asarray(self.probs_fn(record, image_dir), dtype=np.float64)
            fields, tags = fields_from_probs(record, probs)
        else:
            fields, tags = empty_fields(), []
        layout_sec = time.perf_counter() - t0

        timing = dict(timing_in or {})
        timing["layoutxlm"] = round(layout_sec, 3)
        timing["total"] = round(sum(v for k, v in timing.items() if k not in ("total", "features")), 3)
        return {
            "file_id": record["img_id"],
            "fields": fields,
            "n_words": len(record["words"]),
            "skew_angle": skew_angle,
            "bbox_unit": bbox_unit,
            "words": [{"text": w, "bbox": b, "tag": t} for w, b, t in zip(record["words"], boxes, tags)],
            "timing_sec": timing,
        }

    def extract_from_ocr(self, meta: Dict[str, Any], preprocessed_image: str) -> Dict[str, Any]:
        """Fields from an existing OCR result and the preprocessed image it was computed on."""
        image_dir, image_name = os.path.split(os.path.abspath(preprocessed_image))
        record = record_from_ocr(meta, image_name)
        return self._run(record, image_dir, record["pixel_boxes"], "pixel", meta.get("timing_sec"),
                         meta.get("skew_angle"))

    def extract_from_record(self, record: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
        """Fields from a token-dataset record (``words`` and ``boxes`` from the dataset's OCR run, no OCR here).

        ``record["image"]`` is the preprocessed image inside ``image_dir``. Gold labels in the record are ignored.
        """
        rec = {"img_id": record.get("img_id") or record.get("file_id"), "image": record["image"],
               "words": list(record["words"]), "boxes": [list(b) for b in record["boxes"]],
               "labels": ["O"] * len(record["words"]), "gt": {}}
        return self._run(rec, image_dir, rec["boxes"], "normalized_1000", None, None)

    def extract(self, image_path: str) -> Dict[str, Any]:
        """Receipt photo -> fields. Runs the OCR pipeline, then the token classifier on the preprocessed image."""
        if self.ocr_pipeline is None:
            raise RuntimeError("this extractor was built without an OCR pipeline; use extract_from_ocr()")
        import cv2

        meta, image = self.ocr_pipeline.infer(image_path)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{meta['file_id']}.jpg")
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])     # same encoding as the training images
            result = self.extract_from_ocr(meta, path)
        result["image"] = str(image_path)
        return result

    @classmethod
    def from_checkpoint(cls, checkpoint: str, config_dir: str = "configs", device: str = "auto",
                        with_ocr: bool = True) -> "ReceiptExtractor":
        """Load a fine-tuned checkpoint (the ``best`` folder written by training) and, optionally, CRAFT + VietOCR."""
        import torch
        from transformers import AutoProcessor

        from src.common.config import load_config
        from src.training.data import WindowDataset, build_windows
        from src.training.engine import DEFAULTS, load_model, predict_dataset

        cfg = load_config(config_dir)["model"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(cfg["layout_encoder"]["model_name"], apply_ocr=False)
        model = load_model(checkpoint, False, device).eval()

        def probs_fn(record: Dict[str, Any], image_dir: str) -> np.ndarray:
            windows = build_windows([record], processor.tokenizer, DEFAULTS["max_tokens"], DEFAULTS["overlap_tokens"])
            ds = WindowDataset([record], windows, processor, image_dir, DEFAULTS["max_length"])
            merged, _ = predict_dataset(model, ds, [record], device, use_amp=True, batch_size=8, num_workers=0)
            return merged[0]["probs"]

        pipeline = build_ocr_pipeline(cfg, device) if with_ocr else None
        return cls(probs_fn, pipeline)


def build_ocr_pipeline(model_cfg: Dict[str, Any], device: str):
    """Stages 0-2 (preprocessing, CRAFT, VietOCR) from ``configs/model.yaml``; needs torch, craft and vietocr."""
    from src.detection.craft_detector import CraftDetector
    from src.pipeline.runner import ReceiptPipeline
    from src.preprocessing.image_preprocessor import ImagePreprocessor
    from src.recognition.vietocr_recognizer import VietOCRRecognizer

    rec = model_cfg["recognition"]
    return ReceiptPipeline(
        preprocessor=ImagePreprocessor(**model_cfg["preprocessing"]),
        detector=CraftDetector(model_cfg["detection"], device=device),
        recognizer=VietOCRRecognizer(rec, device=device, batch_size=rec["batch_size"]),
        extractor=None,
    )
