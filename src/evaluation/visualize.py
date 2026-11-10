"""Draw a receipt with its gold word labels and a caption of what the model got wrong (Pillow only)."""
import unicodedata
from typing import Any, Dict, Sequence

from PIL import Image, ImageDraw

COLORS = {"SELLER": (30, 136, 229), "ADDRESS": (67, 160, 71), "TIMESTAMP": (251, 140, 0), "TOTAL_COST": (229, 57, 53)}


def ascii_fold(text: Any, limit: int = 90) -> str:
    """Pillow's default font has no Vietnamese letters: drop the diacritics for the caption."""
    s = unicodedata.normalize("NFD", str(text if text is not None else "-").replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s if len(s) <= limit else s[:limit] + "..."


def render_receipt(rec: Dict[str, Any], image: Image.Image, caption_lines: Sequence[str]) -> Image.Image:
    """Image with a coloured box around every word that carries a gold label, under a white caption strip."""
    img = image.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for box, tag in zip(rec["boxes"], rec["labels"]):
        if tag == "O":
            continue
        field = tag.split("-", 1)[1]
        x0, y0, x1, y1 = (box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000)
        draw.rectangle([x0, y0, x1, y1], outline=COLORS.get(field, (0, 0, 0)), width=2)
    line_h = 14
    strip = Image.new("RGB", (max(w, 640), line_h * (len(caption_lines) + 1) + 6), "white")
    sd = ImageDraw.Draw(strip)
    for i, line in enumerate(caption_lines):
        sd.text((4, 3 + i * line_h), ascii_fold(line, 110), fill=(0, 0, 0))
    out = Image.new("RGB", (max(w, 640), strip.height + h), "white")
    out.paste(strip, (0, 0))
    out.paste(img, (0, strip.height))
    return out
