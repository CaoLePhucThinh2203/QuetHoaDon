"""What the web demo shows, as plain Python (no gradio, no torch) so it is unit-tested on any machine.

``scripts/demo_app.py`` turns an uploaded photo into a result with ``ReceiptExtractor`` (the frozen final configuration)
and hands the result to these helpers:

* ``prepare_upload``: rotate the photo by its EXIF orientation and save it as an RGB JPEG (phone photos);
* ``field_rows`` / ``item_rows`` / ``items_summary``: the tables, only with what was read;
* ``draw_words``: the preprocessed image with every OCR word box, coloured by the field the model gave it;
* ``timing_rows``: seconds per stage.

Nothing here changes a prediction: it only formats the output of ``compact_result``.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FIELD_LABELS = {
    "SELLER": "Người bán",
    "ADDRESS": "Địa chỉ",
    "TIMESTAMP": "Ngày giờ",
    "TOTAL_COST": "Tổng tiền",
}
FIELD_ORDER = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")

# BGR colours for OpenCV; the legend in the app uses the same names.
FIELD_COLOURS_BGR = {
    "SELLER": (40, 120, 230),       # orange
    "ADDRESS": (60, 160, 40),       # green
    "TIMESTAMP": (200, 90, 30),     # blue
    "TOTAL_COST": (40, 40, 220),    # red
}
OTHER_COLOUR_BGR = (170, 170, 170)  # words the model left as O

STAGE_LABELS = {
    "preprocess": "Tiền xử lý ảnh",
    "detect": "CRAFT tìm vùng chữ",
    "ocr": "VietOCR đọc chữ",
    "layoutxlm": "LayoutXLM gán nhãn",
    "total": "Tổng",
}


def format_vnd(amount: Optional[int]) -> str:
    """``418986`` -> ``418.986 đ`` (Vietnamese thousands separator)."""
    if amount is None:
        return ""
    return f"{amount:,}".replace(",", ".") + " đ"


def prepare_upload(src_path: str, dst_path: str) -> str:
    """Path of an upright RGB JPEG of the upload: ``src_path`` itself when it already is one, else ``dst_path``.

    An upright RGB JPEG is passed through untouched, so a dataset image goes through the web demo byte for byte as in
    the scored pipeline. Anything else (an EXIF orientation tag from a phone camera, which OpenCV ignores; PNG; alpha;
    grey) is rotated upright and saved as an RGB JPEG at ``dst_path``.
    """
    from PIL import Image, ImageOps

    with Image.open(src_path) as im:
        upright = im.getexif().get(0x0112, 1) == 1
        if upright and im.format == "JPEG" and im.mode == "RGB":
            return src_path
        im = ImageOps.exif_transpose(im)
        im.convert("RGB").save(dst_path, "JPEG", quality=95)
    return dst_path


def field_rows(compact: Dict[str, Any]) -> List[List[str]]:
    """Rows ``[field, value, detail]`` for the fields that were read; a missing field gets no row."""
    rows = []
    fields = compact.get("fields", {})
    for name in FIELD_ORDER:
        f = fields.get(name)
        if not f:
            continue
        detail = ""
        if name == "TIMESTAMP":
            detail = " ".join(x for x in (f.get("date"), f.get("time")) if x)
        elif name == "TOTAL_COST":
            detail = format_vnd(f.get("amount_vnd"))
        rows.append([FIELD_LABELS[name], f["text"], detail])
    return rows


def missing_fields(compact: Dict[str, Any]) -> List[str]:
    fields = compact.get("fields", {})
    return [FIELD_LABELS[n] for n in FIELD_ORDER if n not in fields]


def line_adds_up(item: Dict[str, Any]) -> Optional[bool]:
    """``quantity x unit price == amount`` for one item line; ``None`` when a number is missing."""
    q, p, a = item.get("quantity"), item.get("unit_price"), item.get("amount")
    if q is None or p is None or a is None:
        return None
    return abs(q * p - a) < 1


def item_rows(compact: Dict[str, Any]) -> List[List[Any]]:
    """Rows ``[#, name, quantity, unit price, amount, check]``; empty when no item line was read.

    The check is ``✓`` when quantity x unit price equals the amount, ``⚠`` when it does not (a misread number) and
    empty when a number is missing. It is only shown, the item reader itself is unchanged.
    """
    rows = []
    for i, it in enumerate(compact.get("items") or [], 1):
        ok = line_adds_up(it)
        rows.append([i, it.get("name") or "", it.get("quantity"), format_vnd(it.get("unit_price")),
                     format_vnd(it.get("amount")), "" if ok is None else ("✓" if ok else "⚠")])
    return rows


def items_summary(compact: Dict[str, Any]) -> str:
    """One line (Markdown) on whether the item amounts add up to the total printed on the receipt."""
    check = compact.get("items_check")
    if compact.get("warning") == "unreadable":
        return "⚠️ Ảnh khó đọc: không hiện mặt hàng để tránh số sai."
    if not compact.get("items") or not check:
        return "Không đọc được dòng mặt hàng nào."
    n = len(compact["items"])
    total = check.get("predicted_total")
    line = f"{n} món, tổng các món {format_vnd(check.get('sum'))}"
    if total is not None:
        line += f", tổng tiền đọc được {format_vnd(total)}"
    if check.get("status") == "verified":
        return f"✅ {line}. Tổng các món khớp dòng tổng in trên hóa đơn."
    return f"⚠️ {line}. Tổng các món **không** khớp dòng tổng trên hóa đơn: có thể thiếu hoặc đọc sai một món."


def field_of_tag(tag: Optional[str]) -> Optional[str]:
    """``B-TOTAL_COST`` -> ``TOTAL_COST``; ``O`` or ``None`` -> ``None``."""
    if not tag or tag == "O" or "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def draw_words(image_bgr: np.ndarray, words: Sequence[Dict[str, Any]], thickness: int = 2) -> np.ndarray:
    """Copy of the image with every word box drawn: field colour for tagged words, grey (thin) for the rest.

    ``words`` are the ``result["words"]`` of ``ReceiptExtractor.extract_from_ocr`` (pixel ``[x0, y0, x1, y1]`` on the
    preprocessed image).
    """
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    # grey boxes first so the coloured ones are drawn on top
    ordered = sorted(words, key=lambda wd: field_of_tag(wd.get("tag")) is not None)
    for wd in ordered:
        x0, y0, x1, y1 = (int(round(v)) for v in wd["bbox"][:4])
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
        field = field_of_tag(wd.get("tag"))
        colour = FIELD_COLOURS_BGR.get(field, OTHER_COLOUR_BGR)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, thickness if field else 1)
    return out


def timing_rows(timing: Dict[str, float]) -> List[List[str]]:
    rows = []
    for key in ("preprocess", "detect", "ocr", "layoutxlm", "total"):
        if key in timing:
            rows.append([STAGE_LABELS[key], f"{timing[key]:.2f} s"])
    return rows


def legend_markdown() -> str:
    names = {"SELLER": "cam", "ADDRESS": "xanh lá", "TIMESTAMP": "xanh dương", "TOTAL_COST": "đỏ"}
    parts = [f"**{FIELD_LABELS[n]}**: {names[n]}" for n in FIELD_ORDER]
    return "Màu khung chữ: " + " · ".join(parts) + " · từ không thuộc trường nào: xám."
