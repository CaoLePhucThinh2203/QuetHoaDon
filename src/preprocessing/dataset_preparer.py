"""Reproducible MC-OCR dataset preparation module.

Provides loading, quality filtering, invoice-level splitting, leakage checking,
and manifest export for the MC-OCR dataset.
"""
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sklearn.model_selection import train_test_split

from src.preprocessing.annotation_parser import (
    MCOCRRecord,
    load_and_validate_mcocr_csv,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES = ["train", "validation", "test"]

MANIFEST_COLUMNS = [
    "img_id",
    "image_path",
    "text",
    "label",
    "raw_bbox",
    "normalized_bbox",
]

MANIFEST_SPLIT_COLUMNS = MANIFEST_COLUMNS + ["split"]


class DataLeakageError(ValueError):
    """Raised when data leakage between dataset splits is detected."""
    pass


@dataclass
class ManifestRecord:
    """Represents a single annotation entry in the manifest."""

    img_id: str
    image_path: str
    text: str
    label: str
    raw_bbox: List[int]
    normalized_bbox: List[int]
    split: Optional[str] = None

    def to_dict(self, include_split: bool = False) -> Dict[str, Any]:
        """Convert to dictionary suitable for CSV export."""
        data = {
            "img_id": self.img_id,
            "image_path": self.image_path,
            "text": self.text,
            "label": self.label,
            "raw_bbox": json.dumps(self.raw_bbox),
            "normalized_bbox": json.dumps(self.normalized_bbox),
        }
        if include_split and self.split is not None:
            data["split"] = self.split
        return data


def split_invoices(
    img_ids: List[str],
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministically partition invoice IDs into train, validation, and test splits.

    Matches the Kaggle MC-OCR experiment:
    - First split unique IDs into 80% train and 20% temporary using train_test_split.
    - Then split the temporary set 50/50 into validation and test using train_test_split.

    Args:
        img_ids: Collection of invoice image IDs.
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        Dict mapping split names ('train', 'validation', 'test') to lists of invoice IDs.
    """
    unique_ids = sorted(list(set(img_ids)))
    n_total = len(unique_ids)

    if n_total == 0:
        return {"train": [], "validation": [], "test": []}
    if n_total == 1:
        return {"train": unique_ids, "validation": [], "test": []}

    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=0.2, random_state=random_seed
    )

    if len(temp_ids) >= 2:
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=random_seed
        )
    elif len(temp_ids) == 1:
        val_ids, test_ids = temp_ids, []
    else:
        val_ids, test_ids = [], []

    return {
        "train": list(train_ids),
        "validation": list(val_ids),
        "test": list(test_ids),
    }


def check_data_leakage(
    split_assignments: Dict[str, Union[List[str], Set[str]]],
    expected_img_ids: Optional[Union[List[str], Set[str]]] = None,
) -> bool:
    """Ensure that each img_id belongs to exactly one split.

    Args:
        split_assignments: Mapping from split name to collection of img_ids.
        expected_img_ids: Optional collection of all expected valid invoice IDs.

    Raises:
        DataLeakageError: If any img_id belongs to multiple splits, or if expected
            img_ids are missing or unexpected ones appear.

    Returns:
        True if leakage check passes.
    """
    seen: Dict[str, str] = {}
    leakages: Dict[str, List[str]] = {}

    for split_name, ids in split_assignments.items():
        for img_id in ids:
            if img_id in seen:
                if img_id not in leakages:
                    leakages[img_id] = [seen[img_id]]
                if split_name not in leakages[img_id]:
                    leakages[img_id].append(split_name)
            else:
                seen[img_id] = split_name

    if leakages:
        details = "; ".join(
            f"'{img_id}' present in splits: {splits}" for img_id, splits in sorted(leakages.items())
        )
        raise DataLeakageError(
            f"Data leakage detected! Overlapping invoices found across splits: {details}"
        )

    if expected_img_ids is not None:
        expected_set = set(expected_img_ids)
        assigned_set = set(seen.keys())
        missing = expected_set - assigned_set
        if missing:
            raise DataLeakageError(
                f"Data integrity error: {len(missing)} expected invoice(s) missing from splits: "
                f"{sorted(list(missing))[:5]}"
            )
        unexpected = assigned_set - expected_set
        if unexpected:
            raise DataLeakageError(
                f"Data integrity error: {len(unexpected)} unexpected invoice(s) found in splits: "
                f"{sorted(list(unexpected))[:5]}"
            )

    return True


def check_manifest_leakage(manifest_rows: List[Dict[str, Any]]) -> bool:
    """Ensure that no img_id in manifest rows is associated with more than one split."""
    img_to_splits: Dict[str, Set[str]] = {}
    for row in manifest_rows:
        img_id = row.get("img_id", "")
        split = row.get("split")
        if split:
            img_to_splits.setdefault(img_id, set()).add(split)

    leakages = {
        img_id: sorted(list(splits))
        for img_id, splits in img_to_splits.items()
        if len(splits) > 1
    }
    if leakages:
        details = "; ".join(
            f"'{img_id}': {splits}" for img_id, splits in sorted(leakages.items())
        )
        raise DataLeakageError(
            f"Manifest data leakage detected! Single invoice assigned to multiple splits: {details}"
        )
    return True


def export_manifest_csv(
    manifest_records: List[ManifestRecord],
    output_path: Union[str, Path],
    include_split: bool = False,
) -> None:
    """Write manifest records to a CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    columns = MANIFEST_SPLIT_COLUMNS if include_split else MANIFEST_COLUMNS

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in manifest_records:
            writer.writerow(record.to_dict(include_split=include_split))


def prepare_dataset(
    csv_path: Union[str, Path],
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    random_seed: int = 42,
    scale: int = 1000,
) -> Dict[str, Any]:
    """Prepare a reproducible MC-OCR dataset with clean manifests and invoice-level splits.

    Args:
        csv_path: Path to the raw MC-OCR annotation CSV.
        image_dir: Directory containing receipt images.
        output_dir: Directory to save generated manifests and summary report.
        random_seed: Seed for reproducible invoice shuffling (default: 42).
        scale: Upper coordinate bound for bounding-box normalization (default: 1000).

    Returns:
        Summary dictionary containing dataset statistics and leakage verification.
    """
    csv_file = Path(csv_path)
    images_path = Path(image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and parse raw annotations using annotation_parser
    records, parser_summary = load_and_validate_mcocr_csv(csv_file, scale=scale)
    total_original_records = len(records)

    # 2. Filter records according to requirements:
    #    - is_valid == True
    #    - anno_num > 0
    #    - corresponding image file exists
    valid_records: List[Tuple[MCOCRRecord, Path]] = []
    invalid_records_count = 0
    empty_records_count = 0
    missing_image_records_count = 0

    for rec in records:
        if not rec.is_valid:
            invalid_records_count += 1
            continue
        if rec.anno_num <= 0 or len(rec.instances) == 0:
            empty_records_count += 1
            continue

        img_file = images_path / rec.img_id
        if not (img_file.exists() and img_file.is_file()):
            missing_image_records_count += 1
            continue

        valid_records.append((rec, img_file))

    # 3. Collect unique valid invoice IDs
    unique_valid_ids = sorted(list({rec.img_id for rec, _ in valid_records}))
    valid_invoices_used = len(unique_valid_ids)

    # 4. Split strictly at the invoice/image level
    splits = split_invoices(
        unique_valid_ids,
        random_seed=random_seed,
    )

    # 5. Leakage check ensuring each img_id belongs to exactly one split
    check_data_leakage(splits, expected_img_ids=unique_valid_ids)

    img_id_to_split: Dict[str, str] = {}
    for split_name, ids in splits.items():
        for iid in ids:
            img_id_to_split[iid] = split_name

    # 6. Build clean manifest and split manifest entries
    clean_manifest: List[ManifestRecord] = []
    split_manifest: List[ManifestRecord] = []
    label_distribution: Dict[str, int] = {}
    region_counts_by_split: Dict[str, int] = {name: 0 for name in SPLIT_NAMES}

    for rec, img_file in valid_records:
        split_name = img_id_to_split[rec.img_id]
        for inst in rec.instances:
            clean_rec = ManifestRecord(
                img_id=rec.img_id,
                image_path=str(img_file),
                text=inst.text,
                label=inst.label,
                raw_bbox=inst.raw_bbox,
                normalized_bbox=inst.normalized_bbox,
            )
            clean_manifest.append(clean_rec)

            split_rec = ManifestRecord(
                img_id=rec.img_id,
                image_path=str(img_file),
                text=inst.text,
                label=inst.label,
                raw_bbox=inst.raw_bbox,
                normalized_bbox=inst.normalized_bbox,
                split=split_name,
            )
            split_manifest.append(split_rec)

            label_distribution[inst.label] = label_distribution.get(inst.label, 0) + 1
            region_counts_by_split[split_name] = region_counts_by_split.get(split_name, 0) + 1

    # Verify no manifest-level leakage
    check_manifest_leakage([r.to_dict(include_split=True) for r in split_manifest])

    # 7. Export mcocr_clean_manifest.csv and mcocr_manifest_split.csv
    clean_manifest_path = out_dir / "mcocr_clean_manifest.csv"
    split_manifest_path = out_dir / "mcocr_manifest_split.csv"
    export_manifest_csv(clean_manifest, clean_manifest_path, include_split=False)
    export_manifest_csv(split_manifest, split_manifest_path, include_split=True)

    # 8. Compute split invoice counts
    invoice_counts_by_split = {split_name: len(splits.get(split_name, [])) for split_name in SPLIT_NAMES}

    # 9. Prepare dataset summary
    summary_data = {
        "total_original_records": total_original_records,
        "valid_invoices_used": valid_invoices_used,
        "excluded_empty_records": empty_records_count,
        "invalid_records": invalid_records_count,
        "missing_image_records": missing_image_records_count,
        "annotated_regions": len(clean_manifest),
        "label_distribution": dict(sorted(label_distribution.items())),
        "invoice_counts_by_split": invoice_counts_by_split,
        "region_counts_by_split": region_counts_by_split,
        "leakage_check_result": "PASSED",
    }

    # 10. Export dataset_summary.json
    summary_json_path = out_dir / "dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Prepared dataset: {valid_invoices_used} valid invoices ({len(clean_manifest)} regions) "
        f"exported to {out_dir}"
    )

    return summary_data


# Reusable alias matching the function name requirement
prepare_mcocr_dataset = prepare_dataset
