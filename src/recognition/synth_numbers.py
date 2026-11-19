"""Synthetic receipt-number crops for fine-tuning VietOCR on digits (pure Python + Pillow + OpenCV).

VietOCR was fine-tuned on the lines of the four annotated fields only; item prices, amounts and quantities were never
taught, and most remaining item errors are misread digits. Rendering numbers gives unlimited exactly-labelled crops:
amounts in the receipt styles ("45,000", "210.000", "24 000", "439,850đ", "12,000.00"), weights and quantities ("0.57",
"1,214"), with thermal-print degradations (faded ink, blur, noise, slight rotation, a strike-through). They are mixed
with real crops so the model does not drift away from real photos.

Rows use the format of ``finetune_data`` (``crop``, ``inner``, ``text``, ``kind``, ``label``, ``img_id``).
"""
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts")
FONT_HINTS = ("dejavusans", "dejavusansmono", "liberationsans", "liberationmono", "freesans", "freemono", "arial",
              "consola", "cour", "tahoma", "verdana", "roboto", "notosans", "ubuntu")


def find_fonts(dirs: Sequence[str] = FONT_DIRS) -> List[str]:
    """TrueType fonts on this machine whose name looks like a receipt-printer face (sans or monospace)."""
    out = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            name = os.path.basename(path).lower().replace("-", "").replace("_", "")
            italic = "italic" in name or "oblique" in name or name.endswith(("i.ttf", "z.ttf"))   # arialbi, consolaz
            if any(name.startswith(h) for h in FONT_HINTS) and not italic:
                out.append(path)
    return sorted(set(out))


def _group(value: int, sep: str) -> str:
    return f"{value:,}".replace(",", sep)


def number_text(rng: np.random.Generator) -> str:
    """One random number written the way receipts print it."""
    r = rng.random()
    if r < 0.70:                                   # an amount in VND
        value = int(rng.choice([rng.integers(1, 100), rng.integers(100, 1000), rng.integers(1000, 10000)])) * 500
        if rng.random() < 0.3:
            value += int(rng.integers(0, 500))
        text = _group(value, str(rng.choice([",", ".", " "])))
        tail = rng.random()
        if tail < 0.12:
            text += "đ"
        elif tail < 0.18:
            text += " đ"
        elif tail < 0.24:
            text += ".00" if "," in text else ",00"
        return text
    if r < 0.88:                                   # a weight or decimal quantity
        return f"{int(rng.integers(0, 3))}{rng.choice(['.', ','])}{int(rng.integers(1, 1000)):0{int(rng.integers(2, 4))}d}"
    return str(int(rng.integers(1, 100)))          # a whole quantity


def render_number(text: str, font_path: Optional[str], rng: np.random.Generator) -> Any:
    """BGR crop of ``text`` printed on thermal paper, roughly 24-44 px of text height."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    size = int(rng.integers(22, 40))
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size)
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    pad_x, pad_y = int(rng.integers(4, 14)), int(rng.integers(3, 10))
    w, h = x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y
    paper, ink = int(rng.integers(185, 256)), int(rng.integers(0, 110))
    img = Image.new("L", (w, h), paper)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - x0, pad_y - y0), text, fill=ink, font=font)
    if rng.random() < 0.05:                        # a crossed-out price ("12,000 45,000")
        ym = pad_y + (y1 - y0) // 2
        draw.line([(pad_x, ym), (w - pad_x, ym)], fill=ink, width=max(1, size // 14))
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.4:                         # faded thermal ink: lighten the strokes unevenly
        arr = arr + (paper - arr) * rng.uniform(0.1, 0.5) * rng.random(arr.shape[1])[None, :]
    if rng.random() < 0.4:
        arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        arr = arr + rng.normal(0, rng.uniform(3, 12), arr.shape)
    if rng.random() < 0.5:                         # thermal print: dots of ink missing inside the strokes
        holes = rng.random(arr.shape) < rng.uniform(0.05, 0.3)
        arr = np.where(holes & (arr < (paper + ink) / 2), arr + (paper - arr) * rng.uniform(0.4, 0.9), arr)
    if rng.random() < 0.3:                         # slight rotation, as after imperfect deskew
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.5, 2.5), 1.0)
        arr = cv2.warpAffine(arr, m, (w, h), borderValue=float(paper))
    if rng.random() < 0.6:                         # a small, far-away print: lose resolution, then resize back
        f = rng.uniform(0.35, 0.8)
        small = cv2.resize(arr, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = np.clip(arr, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                         # phone JPEG compression
        ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 75))])
        gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_synth_set(out_dir: str, split: str, n: int, seed: int, fonts: Optional[Sequence[str]] = None,
                    vocab_chars: Optional[str] = None) -> Dict[str, Any]:
    """Write ``<out_dir>/<split>/crops/*.png`` and ``pairs.jsonl`` with ``n`` synthetic numbers (kind ``synth``)."""
    import cv2

    fonts = list(fonts) if fonts is not None else find_fonts()
    rng = np.random.default_rng(seed)
    crop_dir = os.path.join(out_dir, split, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    allowed = set(vocab_chars) if vocab_chars else None
    written = skipped = 0
    with open(os.path.join(out_dir, split, "pairs.jsonl"), "w", encoding="utf-8") as out:
        for i in range(n):
            text = number_text(rng)
            if allowed is not None and not set(text) <= allowed:
                skipped += 1
                continue
            font = fonts[int(rng.integers(len(fonts)))] if fonts else None
            img = render_number(text, font, rng)
            name = f"synth_{split}_{i:06d}.png"
            cv2.imwrite(os.path.join(crop_dir, name), img)
            h, w = img.shape[:2]
            out.write(json.dumps({"crop": f"crops/{name}", "inner": [0, 0, w, h], "text": text, "kind": "synth",
                                  "label": "NUMBER", "img_id": f"synth_{split}_{i // 20:05d}", "ocr": None},
                                 ensure_ascii=False) + "\n")
            written += 1
    return {"split": split, "written": written, "out_of_vocab": skipped, "fonts": len(fonts)}
