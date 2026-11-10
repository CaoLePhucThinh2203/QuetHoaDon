# Image, OCR, and Layout Preprocessing

## 1. Inference data path

The unsupervised inference pipeline starts from receipt images. Ground-truth annotation text and entity labels are not inputs to this path.

1. Load the receipt and record original width/height.
2. Apply necessary image preparation while retaining coordinate transforms.
3. Run CRAFT to obtain text polygons and detection scores.
4. Rectify each polygon and crop the corresponding text region.
5. Run VietOCR on each crop.
6. Attach the recognized text to the original polygon/bounding box.
7. Estimate reading order and prepare LayoutXLM inputs.

## 2. Bounding-box representation

Use pixel coordinates in `[x0, y0, x1, y1]` form internally. Convert CRAFT polygons to axis-aligned boxes for LayoutXLM input, while retaining the original quadrilateral for perspective rectification.

Normalize coordinates to `[0, 1000]`:

`x_norm = round(1000 * x / image_width)`

`y_norm = round(1000 * y / image_height)`

Clamp coordinates to the valid range and enforce `x0 <= x1`, `y0 <= y1`. LayoutXLM subword tokens inherit the bounding box of their parent OCR region. Special tokens use `[0, 0, 0, 0]`.

## 3. Reading order and line grouping

Sort regions using geometry with tolerance for multi-column layouts. A simple global top-to-bottom sort is insufficient for receipts with columns. Group regions into text lines/rows based on vertical overlap and center distance, then order regions left-to-right within each row. Preserve the original order and grouping confidence for later correction.

## 4. Unsupervised grouping features

For each OCR region, derive:
- LayoutXLM contextual embedding;
- normalized box and center;
- text length and character composition;
- numeric/currency/date indicators;
- alignment with adjacent regions;
- relative position within the receipt.

Cluster or link regions using these features. Do not interpret raw cluster IDs as entity names. A separate candidate interpretation layer assigns possible semantic roles from contextual, lexical, geometric, and arithmetic evidence.

## 5. Annotation isolation

`annotation_parser.py` reads MC-OCR annotations for data-quality inspection and evaluation workflows. It is not part of the production/inference data path. Do not use its `text`, `label`, or ground-truth `bbox` values to train LayoutXLM, construct pseudo-labels, tune clustering thresholds, or correct OCR output in an experiment described as unsupervised.

For a strict unsupervised experiment, freeze algorithm parameters before inspecting held-out ground-truth results. Report evaluation as a separate measurement step.

## 6. Quality handling

- Keep detection score and OCR confidence as separate fields.
- If the OCR library does not provide a calibrated confidence, use `null`.
- Keep multiple candidates when a field is ambiguous.
- Never fabricate missing seller, address, item, or total values.
- Mark uncertain or missing outputs explicitly.
