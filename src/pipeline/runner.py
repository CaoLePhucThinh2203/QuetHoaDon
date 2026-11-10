"""Batch runner: preprocess -> detect -> recognize -> (optional) LayoutXLM features.

Per image it writes ``<id>.json`` (OCR result + metadata) and ``<id>.pt`` (features).
The JSON is written last and atomically, so *JSON exists* means *image finished* and a
killed Colab session can simply be restarted.
"""
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class OutputDirs:
    base: str
    images: str
    ocr_json: str
    features: str

    @classmethod
    def from_base(cls, base: str) -> "OutputDirs":
        d = cls(base, os.path.join(base, "preprocessed_images"),
                os.path.join(base, "ocr_annotations"), os.path.join(base, "layoutxlm_features"))
        for p in (d.images, d.ocr_json, d.features):
            os.makedirs(p, exist_ok=True)
        return d

    def paths(self, file_id: str) -> Tuple[str, str, str]:
        return (os.path.join(self.ocr_json, f"{file_id}.json"),
                os.path.join(self.features, f"{file_id}.pt"),
                os.path.join(self.images, f"{file_id}.jpg"))


def _atomic_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _default_save_tensor(obj: Any, path: str) -> None:
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class ReceiptPipeline:
    """Stages 0-3 of the receipt pipeline: preprocess -> detect (CRAFT) -> recognise (VietOCR) -> features.

    ``predict()`` handles one image in memory (used by ``scripts/run_pipeline.py``); ``run()`` handles a dataset
    with checkpointing. ``extractor=None`` stops after OCR (no LayoutXLM / detectron2 needed).
    """

    def __init__(self, preprocessor, detector, recognizer, extractor, dirs: Optional[OutputDirs] = None,
                 save_tensor: Callable[[Any, str], None] = _default_save_tensor) -> None:
        self.preprocessor = preprocessor
        self.detector = detector
        self.recognizer = recognizer
        self.extractor = extractor        # may be None -> OCR only
        self.dirs = dirs                  # only needed by run() / process_one()
        self._save_tensor = save_tensor

    # ---- checkpointing -------------------------------------------------------------
    def is_done(self, file_id: str) -> bool:
        json_path, pt_path, _ = self.dirs.paths(file_id)
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        if meta.get("n_text_regions", meta.get("total_words", -1)) == 0:
            return True                   # receipt without text: nothing to extract
        return os.path.exists(pt_path) or self.extractor is None

    # ---- one image -----------------------------------------------------------------
    def _infer(self, img_path: Union[str, Path], file_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any]:
        """Run stages 0-3 in memory. Returns ``(meta, features or None, preprocessed BGR image)``."""
        import cv2
        from PIL import Image

        t0 = time.time()
        pre = self.preprocessor.process(str(img_path))
        h, w = pre.image.shape[:2]
        t1 = time.time()

        regions = self.detector.detect(pre.image, scale_ratio=pre.scale)
        t2 = time.time()
        items = self.recognizer.recognize_regions(pre.image, regions) if regions else []
        t3 = time.time()

        meta: Dict[str, Any] = {
            "file_id": file_id,
            "orig_shape": list(pre.original.shape[:2]),
            "processed_shape": [h, w, 3],
            "scale_factor": float(pre.scale),
            "skew_angle": float(pre.skew_angle),
            "n_detected": len(regions),
            "n_text_regions": len(items),
            "ocr_items": [it.to_dict() for it in items],
        }

        feats = None
        if items and self.extractor is not None:
            feats = self.extractor.extract(Image.fromarray(cv2.cvtColor(pre.image, cv2.COLOR_BGR2RGB)), items)
            meta["possibly_truncated"] = bool(feats.get("possibly_truncated", False))
        t4 = time.time()

        meta["timing_sec"] = {"preprocess": round(t1 - t0, 3), "detect": round(t2 - t1, 3),
                              "ocr": round(t3 - t2, 3), "features": round(t4 - t3, 3),
                              "total": round(t4 - t0, 3)}
        return meta, feats, pre.image

    def predict(self, img_path: Union[str, Path]) -> Dict[str, Any]:
        """Run the pipeline on one image without writing anything to disk.

        Args:
            img_path: path to a receipt image (jpg/png).

        Returns:
            The OCR result dictionary (same schema as ``<id>.json``, see ``docs/architecture.md`` section 3):
            ``file_id``, shapes, ``skew_angle``, ``n_detected``, ``n_text_regions``, ``timing_sec`` and
            ``ocr_items`` (``text``, ``ocr_confidence``, ``bbox`` ... in reading order).
        """
        return self._infer(img_path, Path(img_path).stem)[0]

    def process_one(self, img_path: Union[str, Path], file_id: str) -> Dict[str, int]:
        """Process one image and write ``<id>.jpg``, ``<id>.pt`` (if features) and ``<id>.json`` (last)."""
        import cv2

        json_path, pt_path, img_out = self.dirs.paths(file_id)
        meta, feats, image = self._infer(img_path, file_id)
        cv2.imwrite(img_out, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if feats is not None and feats.get("last_hidden_state") is not None:
            self._save_tensor({k: feats[k] for k in (
                "last_hidden_state", "input_ids", "normalized_bboxes", "attention_mask", "words")
                } | {"file_id": file_id}, pt_path)
        _atomic_json(meta, json_path)     # last: marks the image as finished
        return {"detected": meta["n_detected"], "ocr": meta["n_text_regions"]}

    # ---- many images ---------------------------------------------------------------
    def run(self, paths: List[Union[str, Path]], abort_after: int = 20, abort_fail_ratio: float = 0.5,
            show_errors: int = 3, progress: bool = True) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        """Process many images, writing one result set per image into ``self.dirs``.

        Already finished images are skipped (resume after a disconnect). The run stops early if more than
        ``abort_fail_ratio`` of the first ``abort_after``+ attempted images fail, so a systematic bug shows up
        in seconds instead of after the whole dataset.

        Args:
            paths: image paths; the file stem is used as ``file_id``.
            abort_after: minimum number of attempted images before the early-stop rule applies.
            abort_fail_ratio: failure ratio above which the run stops.
            show_errors: number of tracebacks printed during the run (all are saved to ``failed.json``).
            progress: show a tqdm progress bar.

        Returns:
            ``(stats, failed)``: counts ``{"ok", "empty", "skipped", "failed"}`` and ``[(file_id, traceback)]``.
        """
        stats = {"ok": 0, "empty": 0, "skipped": 0, "failed": 0}
        failed: List[Tuple[str, str]] = []
        t0 = time.time()

        iterator: Any = paths
        write = print
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(paths, desc="Processing")
            write = tqdm.write

        for p in iterator:
            file_id = Path(p).stem
            if self.is_done(file_id):
                stats["skipped"] += 1
                continue
            try:
                r = self.process_one(p, file_id)
                stats["ok" if r["ocr"] > 0 else "empty"] += 1
            except Exception:
                tb = traceback.format_exc()
                failed.append((file_id, tb))
                stats["failed"] += 1
                if len(failed) <= show_errors:
                    write(f"[!] {file_id}\n{tb}")

            tried = stats["ok"] + stats["empty"] + stats["failed"]
            if tried >= abort_after and stats["failed"] / tried > abort_fail_ratio:
                write(f"[x] Early stop: {stats['failed']}/{tried} images failed. Fix the error above and rerun.")
                break
            if progress:
                iterator.set_postfix(stats)

        if failed:
            _atomic_json([{"file_id": a, "traceback": b} for a, b in failed],
                         os.path.join(self.dirs.base, "failed.json"))
        print(f"\n[done] {stats} | {time.time() - t0:.0f}s | OCR crop errors: {getattr(self.recognizer, 'n_errors', 0)}")
        return stats, failed
