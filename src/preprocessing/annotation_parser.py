"""Parser and validator for MC-OCR dataset annotations."""
import ast
import csv
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.bbox_utils import normalize_bbox

logger = logging.getLogger(__name__)

# Known label normalizations
LABEL_NORMALIZATION = {
    "SELLER": "SELLER",
    "ADDRESS": "ADDRESS",
    "TIMESTAMP": "TIMESTAMP",
    "TOTAL_COST": "TOTAL_COST",
    "TOTAL_TOTAL_COST": "TOTAL_COST",  # Known anomaly in raw dataset
}

CATEGORY_ID_TO_LABEL = {
    15: "SELLER",
    16: "ADDRESS",
    17: "TIMESTAMP",
    18: "TOTAL_COST",
}


@dataclass
class AnnotationInstance:
    """Represents a single text annotation on a receipt."""
    text: str
    label: str
    category_id: int
    raw_bbox: List[int]  # [x, y, w, h]
    normalized_bbox: List[int]  # [x0, y0, x1, y1] in [0, 1000]
    segmentation: List[Any]
    area: float
    width: int
    height: int


@dataclass
class MCOCRRecord:
    """Represents a full receipt image and its associated annotations."""
    img_id: str
    anno_num: int
    image_quality: float
    instances: List[AnnotationInstance] = field(default_factory=list)
    is_valid: bool = True
    error_message: Optional[str] = None


def parse_raw_polygons(raw_str: str) -> List[Dict[str, Any]]:
    """Safely parse the anno_polygons string from CSV using ast or json."""
    if not raw_str or raw_str.strip() in ("", "[]"):
        return []

    # Try json first, fallback to ast.literal_eval
    try:
        data = json.loads(raw_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    try:
        data = ast.literal_eval(raw_str)
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning(f"Failed to parse polygons string: {e}")
        return []

    return []


def parse_mcocr_row(row: Dict[str, str], scale: int = 1000) -> MCOCRRecord:
    """Parse a single row from mcocr_train_df.csv into a validated MCOCRRecord."""
    img_id = row.get("img_id", "").strip()
    quality = float(row.get("anno_image_quality", 0.0) or 0.0)

    try:
        anno_num = int(row.get("anno_num", 0) or 0)
    except ValueError:
        anno_num = 0

    if anno_num == 0:
        return MCOCRRecord(
            img_id=img_id,
            anno_num=0,
            image_quality=quality,
            instances=[],
            is_valid=True,
        )

    raw_texts = row.get("anno_texts", "")
    raw_labels = row.get("anno_labels", "")
    raw_polys = row.get("anno_polygons", "")

    texts = raw_texts.split("|||") if raw_texts else []
    labels = raw_labels.split("|||") if raw_labels else []
    polygons = parse_raw_polygons(raw_polys)

    # Check structural alignment
    n_texts = len(texts)
    n_labels = len(labels)
    n_polys = len(polygons)

    if not (n_texts == n_labels == n_polys):
        err = f"Length mismatch for {img_id}: texts={n_texts}, labels={n_labels}, polys={n_polys}"
        return MCOCRRecord(
            img_id=img_id,
            anno_num=anno_num,
            image_quality=quality,
            instances=[],
            is_valid=False,
            error_message=err,
        )

    instances: List[AnnotationInstance] = []
    for text, label, poly in zip(texts, labels, polygons):
        clean_label = LABEL_NORMALIZATION.get(label.strip(), label.strip())
        cat_id = poly.get("category_id", -1)
        raw_bbox = poly.get("bbox", [0, 0, 0, 0])
        seg = poly.get("segmentation", [])
        area = float(poly.get("area", 0.0))
        w = int(poly.get("width", 1000) or 1000)
        h = int(poly.get("height", 1000) or 1000)

        norm_box = normalize_bbox(raw_bbox, width=w, height=h, scale=scale, is_xywh=True)

        instances.append(
            AnnotationInstance(
                text=text.strip(),
                label=clean_label,
                category_id=cat_id,
                raw_bbox=raw_bbox,
                normalized_bbox=norm_box,
                segmentation=seg,
                area=area,
                width=w,
                height=h,
            )
        )

    return MCOCRRecord(
        img_id=img_id,
        anno_num=len(instances),
        image_quality=quality,
        instances=instances,
        is_valid=True,
    )


def load_and_validate_mcocr_csv(
    csv_path: Union[str, Path],
    scale: int = 1000,
) -> Tuple[List[MCOCRRecord], Dict[str, Any]]:
    """Load an MC-OCR CSV file and generate a data quality report."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_file}")

    records: List[MCOCRRecord] = []
    label_counts: Dict[str, int] = {}
    invalid_records: List[str] = []
    empty_records: List[str] = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = parse_mcocr_row(row, scale=scale)
            records.append(record)

            if not record.is_valid:
                invalid_records.append(f"{record.img_id}: {record.error_message}")
            elif record.anno_num == 0:
                empty_records.append(record.img_id)
            else:
                for inst in record.instances:
                    label_counts[inst.label] = label_counts.get(inst.label, 0) + 1

    summary = {
        "total_records": len(records),
        "valid_records": len([r for r in records if r.is_valid]),
        "invalid_records_count": len(invalid_records),
        "invalid_records_details": invalid_records,
        "empty_records_count": len(empty_records),
        "empty_records_img_ids": empty_records,
        "label_distribution": label_counts,
    }

    return records, summary
