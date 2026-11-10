# Dataset Documentation

## MC-OCR 2021 — Primary Vietnamese receipt dataset

The repository configuration identifies MC-OCR 2021 as the primary dataset. Its CSV annotations include image identifiers, text strings, entity labels, polygons, and image-quality metadata.

The annotation parser supports audit tasks such as deserialization, structural consistency checks, normalization of a known label spelling anomaly, and reporting empty records. These labels are **not used to train or fine-tune the unsupervised extraction pipeline**.

### Permitted use in this project
- inspect dataset structure and image readability;
- validate parser behavior and annotation consistency;
- evaluate the final frozen pipeline against ground truth;
- report metrics after the algorithm and thresholds have been fixed.

### Prohibited use for an unsupervised claim
- training a LayoutXLM token-classification head from `anno_labels`;
- using ground-truth text/boxes to supervise CRAFT or OCR;
- generating pseudo-labels directly from ground-truth entity labels;
- tuning thresholds repeatedly against the evaluation labels and then describing the result as fully unsupervised.

The dataset's exact local availability and counts must be rechecked in the execution environment. Do not treat historical report figures as a fresh verification.

## Other datasets

SROIE 2019 (English) and CORD (Indonesian) are listed as possible future datasets. Their integration status must be updated only after the files have actually been obtained and inspected. VietOCR is Vietnamese-oriented; other languages may require a language-appropriate recognition model.

## Dataset/evaluation separation

Maintain separate directories and data loaders for:
- raw receipt images used by inference;
- annotation files reserved for audit/evaluation;
- cached CRAFT detections and VietOCR outputs;
- final structured predictions.

Do not mix evaluation labels into cached inference features.
