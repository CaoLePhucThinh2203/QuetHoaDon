"""(Crop, number) pairs from real receipts whose item lines are ``verified`` (pure Python + OpenCV).

A receipt is ``verified`` when the item amounts read by ``line_items`` add up to an amount printed on one of its own
"total" lines; on train that sum equals the annotated total on 97 % of such receipts, so the unit prices, amounts and
totals of those receipts are trusted values. Every OCR region (CRAFT box) whose number is one of them becomes a pair:

* ``number``: one box read alone, as the pipeline reads it. Its label is the OCR text itself (the value checks it), so
  these pairs mostly keep the model from forgetting item numbers while it learns from synthetic ones;
* ``number_split``: two overlapping boxes of one number that OCR read with the shared character twice ("399" + "9,000")
  with the joined text as label ("399,000"). Only useful when the pipeline reads merged segments
  (``detection.merge_lines``), because then it sees that crop; train with ``--kinds`` accordingly.

Rows use the format of ``finetune_data`` (``crop``, ``inner``, ``text``, ``kind``, ``label``, ``img_id``, ``ocr``).
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set

from tqdm.auto import tqdm

from src.extraction.line_items import check_items, extract_items, parse_amounts
from src.preprocessing.token_dataset import items_to_words, read_split_manifest
from src.recognition.finetune_data import crop_with_margin

_NUMERIC = set("0123456789.,")


def trusted_values(words: Sequence[str], boxes: Sequence[Sequence[float]]) -> Set[int]:
    """Unit prices, amounts and printed totals of a ``verified`` receipt; empty when the items are not verified."""
    read = extract_items(words, boxes)
    check = check_items(read["items"], read["summary_amounts"])
    if check["status"] != "verified":
        return set()
    values = {int(it["amount"]) for it in read["items"] if it["amount"] > 0}
    values |= {int(it["unit_price"]) for it in read["items"] if isinstance(it["unit_price"], int) and it["unit_price"] > 0}
    return values | {int(t) for t in check["totals"] if t}


def _is_number(text: str) -> bool:
    return bool(text) and set(text) <= _NUMERIC and any(c.isdigit() for c in text)


def _value(text: str) -> Optional[int]:
    amounts = parse_amounts(text)
    return amounts[0][0] if len(amounts) == 1 and amounts[0][1] == 0 and amounts[0][2] == len(text) else None


def receipt_number_pairs(meta: Dict[str, Any], max_single: int = 12) -> List[Dict[str, Any]]:
    """``number`` and ``number_split`` pairs of one receipt (boxes in processed-image pixels)."""
    from src.pipeline.extractor import record_from_ocr

    rec = record_from_ocr(meta, "x.jpg")
    trusted = trusted_values(rec["words"], rec["boxes"])
    if not trusted:
        return []
    words = [w for w in items_to_words(meta.get("ocr_items") or []) if _is_number(w["text"])]
    pairs: List[Dict[str, Any]] = []
    used: Set[int] = set()
    for i, a in enumerate(words):                  # overlapping halves of one number
        for j, b in enumerate(words):
            if i == j or i in used or j in used:
                continue
            ba, bb = a["bbox"], b["bbox"]
            overlap_v = min(ba[3], bb[3]) - max(ba[1], bb[1])
            if not (ba[0] < bb[0] < ba[2] <= bb[2]) or overlap_v <= 0.5 * min(ba[3] - ba[1], bb[3] - bb[1]):
                continue
            ta, tb = a["text"], b["text"]
            if ta[-1] != tb[0] or not (tb[0] in ".," or (len(tb) > 1 and tb[1] in ".,")):
                continue
            joined = ta + tb[1:]
            if _value(joined) in trusted:
                box = [min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])]
                pairs.append({"kind": "number_split", "text": joined, "bbox": box, "ocr": f"{ta} {tb}"})
                used |= {i, j}
    singles = 0
    for i, w in enumerate(words):                  # numbers read alone that the arithmetic confirms
        if i in used or singles >= max_single:
            continue
        if _value(w["text"]) in trusted:
            pairs.append({"kind": "number", "text": w["text"], "bbox": list(w["bbox"]), "ocr": w["text"]})
            singles += 1
    return pairs


def build_number_set(ocr_dir: str, image_dir: str, split_manifest: str, splits: Sequence[str], out_dir: str,
                     vocab_chars: Optional[str] = None, max_single: int = 12) -> Dict[str, Any]:
    """Write ``<out_dir>/<split>/crops/*.png`` and ``pairs.jsonl`` from an OCR run (JSON + processed images)."""
    import cv2

    split_of = read_split_manifest(split_manifest)
    allowed = set(vocab_chars) if vocab_chars else None
    report: Dict[str, Any] = {}
    for split in splits:
        crop_dir = os.path.join(out_dir, split, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        counts = {"receipts": 0, "verified_receipts": 0, "number": 0, "number_split": 0, "missing": 0, "skipped": 0}
        with open(os.path.join(out_dir, split, "pairs.jsonl"), "w", encoding="utf-8") as out:
            for img_id in tqdm(sorted(i for i, s in split_of.items() if s == split), desc=f"numbers {split}"):
                file_id = os.path.splitext(img_id)[0]
                json_path = os.path.join(ocr_dir, f"{file_id}.json")
                image = cv2.imread(os.path.join(image_dir, f"{file_id}.jpg")) if os.path.exists(json_path) else None
                if image is None:
                    counts["missing"] += 1
                    continue
                with open(json_path, encoding="utf-8") as f:
                    pairs = receipt_number_pairs(json.load(f), max_single)
                counts["receipts"] += 1
                counts["verified_receipts"] += bool(pairs)
                for n, p in enumerate(pairs):
                    crop, inner = crop_with_margin(image, p["bbox"])
                    if (allowed is not None and not set(p["text"]) <= allowed) or inner[2] - inner[0] < 4 \
                            or inner[3] - inner[1] < 4:
                        counts["skipped"] += 1
                        continue
                    name = f"{file_id}_{n:03d}_{p['kind']}.png"
                    cv2.imwrite(os.path.join(crop_dir, name), crop)
                    out.write(json.dumps({"crop": f"crops/{name}", "inner": inner, "text": p["text"], "kind": p["kind"],
                                          "label": "NUMBER", "img_id": img_id, "ocr": p["ocr"]},
                                         ensure_ascii=False) + "\n")
                    counts[p["kind"]] += 1
        report[split] = counts
    with open(os.path.join(out_dir, "number_pairs_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
