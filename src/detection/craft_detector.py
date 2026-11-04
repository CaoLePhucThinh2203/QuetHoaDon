"""CRAFT text detector (pretrained, inference only) built on ``craft-text-detector``.

Why the patching below: the library's default ``poly=True`` returns polygons with a varying
number of points and then calls ``np.array(...)`` on that ragged list, which raises
``setting an array element with a sequence ... inhomogeneous shape`` on modern NumPy.
The pipeline only needs boxes, so we force 4-point boxes (``poly=False``).
"""
from typing import List, Optional

import numpy as np

from src.detection.detector import BaseTextDetector, TextRegion
from src.detection.postprocess import drop_nested, sort_reading_order

_PATCHED = False


def _safe_adjust(polys, ratio_w=1, ratio_h=1, ratio_net=2):
    if polys is None or len(polys) == 0:
        return polys
    return [None if p is None else np.array(p, dtype=np.float32) * (ratio_w * ratio_net, ratio_h * ratio_net)
            for p in polys]


def patch_craft_library() -> None:
    """Idempotently patch craft-text-detector and torchvision incompatibilities."""
    global _PATCHED
    if _PATCHED:
        return
    import torchvision.models.vgg as vgg
    if not hasattr(vgg, "model_urls"):   # removed in recent torchvision
        vgg.model_urls = {
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
        }
    import craft_text_detector.craft_utils as craft_utils
    import craft_text_detector.predict as craft_predict

    orig = craft_utils.getDetBoxes

    def no_poly(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, *a, **k):
        return orig(textmap, linkmap, text_threshold, link_threshold, low_text, False)

    for mod in (craft_utils, craft_predict):   # the symbol may be imported into either module
        if hasattr(mod, "adjustResultCoordinates"):
            mod.adjustResultCoordinates = _safe_adjust
        if hasattr(mod, "getDetBoxes"):
            mod.getDetBoxes = no_poly
    _PATCHED = True


class CraftDetector(BaseTextDetector):
    def __init__(self, config: Optional[dict] = None, device: str = "cuda") -> None:
        super().__init__(config)
        patch_craft_library()
        from craft_text_detector import Craft

        c = self.config
        base = dict(output_dir=None, cuda=(device == "cuda"),
                    text_threshold=c.get("text_threshold", 0.7),
                    link_threshold=c.get("link_threshold", 0.4),
                    low_text=c.get("low_text", 0.4))
        try:   # no refine-net; long_size matches the Stage-0 resize
            self.detector = Craft(**base, refiner=False, long_size=c.get("long_size", 1280), export_extra=False)
        except TypeError:   # other library version
            self.detector = Craft(**base, crop_type=None)
            if hasattr(self.detector, "refine_net"):
                self.detector.refine_net = None
        self.nested_thr = c.get("nested_threshold", 0.8)

    def detect(self, image_bgr: np.ndarray, scale_ratio: float = 1.0) -> List[TextRegion]:
        import cv2
        h_img, w_img = image_bgr.shape[:2]
        prediction = self.detector.detect_text(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        raw = prediction.get("boxes", [])
        if raw is None or len(raw) == 0:
            return []

        regions: List[TextRegion] = []
        for box in raw:
            pts = np.asarray(box, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                continue
            x0 = int(max(0, np.floor(pts[:, 0].min())))
            y0 = int(max(0, np.floor(pts[:, 1].min())))
            x1 = int(min(w_img, np.ceil(pts[:, 0].max())))
            y1 = int(min(h_img, np.ceil(pts[:, 1].max())))
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                continue
            regions.append(TextRegion(
                polygon=pts.astype(int).tolist(),
                bbox=[x0, y0, x1, y1],
                orig_bbox=[int(v / scale_ratio) for v in (x0, y0, x1, y1)],
                score=None,   # craft-text-detector does not expose a per-box score
            ))
        return sort_reading_order(drop_nested(regions, self.nested_thr))
