"""VietOCR (vgg_transformer) recognizer - pretrained, inference only."""
from typing import List, Optional

import numpy as np

from src.recognition.recognizer import BaseTextRecognizer, TextRecognitionResult


class VietOCRRecognizer(BaseTextRecognizer):
    def __init__(self, config: Optional[dict] = None, device: str = "cuda", batch_size: int = 32) -> None:
        super().__init__(config, batch_size)
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        c = self.config
        cfg = Cfg.load_config_from_name(c.get("backbone", "vgg_transformer"))
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = c.get("beamsearch", False)
        self.predictor = Predictor(cfg)

    def recognize_batch(self, crops_rgb: List[np.ndarray]) -> List[TextRecognitionResult]:
        from PIL import Image
        pil = [Image.fromarray(c) for c in crops_rgb]
        try:   # fast path: bucketed batch inference
            texts, probs = self.predictor.predict_batch(pil, return_prob=True)
            return [TextRecognitionResult(t, float(p)) for t, p in zip(texts, probs)]
        except Exception:   # fallback: one crop at a time
            out = []
            for img in pil:
                try:
                    t, p = self.predictor.predict(img, return_prob=True)
                    out.append(TextRecognitionResult(t, float(p)))
                except Exception:
                    self.n_errors += 1
                    out.append(TextRecognitionResult("", 0.0))
            return out
