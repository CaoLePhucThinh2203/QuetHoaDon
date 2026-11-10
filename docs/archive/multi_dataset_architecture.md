# Multi-Dataset Architecture and Language Strategy

## 1. Current scope

MC-OCR 2021 is the primary Vietnamese receipt dataset. SROIE 2019 and CORD are future integration candidates, not assumed to be present or evaluated.

## 2. Dataset adapters

Each dataset adapter should convert image paths and metadata into a common image-level input contract. Annotation adapters are separate and must only be enabled for audit/evaluation. The unsupervised inference pipeline consumes images, not entity labels.

## 3. Language-specific OCR

CRAFT detects text regions independent of the semantic language in the detected pixels, but OCR recognition is language/model dependent. VietOCR is selected for Vietnamese receipts. For English or Indonesian receipts, evaluate a suitable recognizer for that language rather than assuming VietOCR transfers reliably.

## 4. Shared extraction representation

The common representation includes OCR text, original page coordinates, normalized bounding boxes, detection/OCR quality signals, reading order, and LayoutXLM contextual features. Unsupervised grouping can be shared only after language-specific OCR and text-pattern handling are validated.

## 5. Entity schema

The target output is:
- `SELLER`: merchant or seller name;
- `ADDRESS`: merchant address/location;
- `ITEM_NAME`: purchased item description;
- `ITEM_PRICE`: item-level price (unit or line price must be distinguished);
- `TOTAL_COST`: final receipt total.

Optional `QUANTITY` and `TIMESTAMP` fields may be retained when detected. Dataset-specific labels must not be forced into unsupported canonical fields.

## 6. Cross-dataset evaluation

Evaluate each language/domain separately. Report OCR quality and field extraction quality separately. A model working on Vietnamese MC-OCR does not establish performance on English SROIE or Indonesian CORD.
