# Project Progress Report — Unsupervised Receipt Extraction

## Project direction

The project direction is an inference-first unsupervised pipeline for Vietnamese receipt understanding:

1. CRAFT detects text regions and their 2D positions.
2. Each detected region is rectified/cropped and recognized by VietOCR.
3. LayoutXLM produces contextual and layout-aware representations from OCR text and bounding boxes.
4. Unsupervised grouping and a separate candidate-interpretation layer identify seller, address/location, item names, item prices, and total amount.
5. The system exports structured results with uncertainty indicators.

## Methodological definition

No receipt-specific entity labels are used to train a task-specific extraction classifier. The upstream models are pretrained models and may have been trained with supervision by their original developers. This project uses their pretrained weights for detection, recognition, and feature extraction; it does not claim that the underlying foundation models were originally trained unsupervised.

LayoutXLM is not a ready-made unsupervised entity extractor. It returns representations. Clustering alone cannot name clusters as `SELLER`, `ITEM_PRICE`, or `TOTAL_COST`; semantic interpretation requires additional spatial, lexical, structural, and arithmetic rules.

## Intended output fields

- Seller / merchant name
- Seller address or location
- Each item name
- Each item's price, distinguishing unit price from line amount where possible
- Receipt total
- Optional quantity and transaction timestamp when evidence is available

## Existing repository components

- Bounding-box conversion, normalization, denormalization, and IoU utilities.
- Abstract CRAFT detector and VietOCR recognizer interfaces.
- MC-OCR annotation parser for audit and evaluation only.
- Extraction field schema and YAML configurations.
- Unit tests for geometry, annotation parsing, and schema mapping.

## Work not yet verified as implemented

The current repository does not by itself establish a working end-to-end system. Concrete CRAFT and VietOCR wrappers, crop rectification, LayoutXLM encoder inference, unsupervised clustering, candidate assignment, output serialization, and end-to-end evaluation still require implementation and testing.

## Evaluation protocol

Freeze the inference pipeline before using MC-OCR labels for evaluation. Report:
- CRAFT detection recall/precision where box annotations are suitable;
- OCR character/word error rate;
- field-level exact match or precision/recall/F1 for seller, address, and total;
- item-row and item-price association metrics;
- end-to-end receipt-level accuracy;
- coverage and uncertainty rate.

Ground-truth annotations must not be used to train or tune a method that is reported as unsupervised.
