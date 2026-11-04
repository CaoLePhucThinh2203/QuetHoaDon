"""Stage 0 - image preprocessing: resize, light deskew, optional contrast enhancement."""
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original: np.ndarray   # BGR, untouched
    image: np.ndarray      # BGR, resized (+ deskewed / enhanced)
    scale: float           # processed_size / original_size (before rotation)
    skew_angle: float      # degrees applied (cv2.getRotationMatrix2D convention), 0.0 if none


class ImagePreprocessor:
    """Resize keeping aspect ratio -> deskew (projection profile) -> optional CLAHE + bilateral."""

    def __init__(self, target_max_dim: int = 1280, min_skew: float = 0.5,
                 max_skew: float = 15.0, enhance: bool = True) -> None:
        self.target_max_dim = target_max_dim
        self.min_skew = min_skew
        self.max_skew = max_skew
        self.enhance = enhance

    def estimate_skew(self, gray: np.ndarray) -> float:
        """Angle (degrees) that makes text lines most horizontal; 0.0 if no clear gain."""
        h, w = gray.shape
        s = 640.0 / max(h, w)
        small = (cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                 if s < 1 else gray)
        bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        hh, ww = bw.shape
        center = (ww // 2, hh // 2)

        def score(angle: float) -> float:
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            r = cv2.warpAffine(bw, m, (ww, hh), flags=cv2.INTER_NEAREST)
            prof = r.sum(axis=1, dtype=np.float64)
            return float(np.sum(np.diff(prof) ** 2))

        coarse = np.arange(-self.max_skew, self.max_skew + 1e-6, 1.0)
        best = max(coarse, key=score)
        best = max(np.arange(best - 1.0, best + 1.0 + 1e-6, 0.25), key=score)
        return float(best) if score(best) > 1.05 * score(0.0) else 0.0

    def process(self, image_path: str) -> PreprocessResult:
        orig = cv2.imread(image_path)
        if orig is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h0, w0 = orig.shape[:2]
        scale = 1.0
        if max(h0, w0) > self.target_max_dim:
            scale = self.target_max_dim / float(max(h0, w0))
            img = cv2.resize(orig, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = orig.copy()

        angle = 0.0
        a = self.estimate_skew(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if self.min_skew <= abs(a) <= self.max_skew:
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            angle = a

        if self.enhance:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a_, b_ = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            img = cv2.cvtColor(cv2.merge((l, a_, b_)), cv2.COLOR_LAB2BGR)
            img = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

        return PreprocessResult(original=orig, image=img, scale=scale, skew_angle=angle)
