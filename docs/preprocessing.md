# Preprocessing & Data Preparation Specification

## 1. Raw Annotation Parsing

The training dataset file `mcocr_train_df.csv` contains structured annotations with custom serialization formats. The preprocessing pipeline applies the following extraction logic:

### Safe Deserialization
1. **Empty Row Handling:** Rows where `anno_num == 0` (such as index 1011 and 1091) have empty string fields for text and labels, and `'[]'` for polygons. These must be parsed into empty lists without triggering index errors.
2. **Polygons:** Deserialized using `ast.literal_eval`. Each entry is a COCO-style dictionary:
   - `bbox`: `[x, y, width, height]` (pixel coordinates)
   - `segmentation`: `[[x1, y1, x2, y2, ...]]` (polygon vertex points)
   - `width`, `height`: Reference image dimensions
3. **Texts & Labels:** Deserialized by splitting on delimiter `|||`.
   - Length assertion: $\text{len}(texts) == \text{len}(labels) == \text{len}(polygons)$.
4. **Label Normalization:** Map known label anomalies:
   - `"TOTAL_TOTAL_COST"` $\to$ `"TOTAL_COST"`

---

## 2. Geometric Coordinate Processing

LayoutXLM requires integer bounding boxes scaled to $[0, 1000]$ relative to the original image dimensions $(W, H)$:

```python
def normalize_bbox(bbox, width, height, scale=1000):
  """Converts [x, y, w, h] in image pixels to [x0, y0, x1, y1] normalized to [0, scale]."""
  x, y, w, h = bbox
  x0 = max(0, min(scale, int(round((x / width) * scale))))
  y0 = max(0, min(scale, int(round((y / height) * scale))))
  x1 = max(0, min(scale, int(round(((x + w) / width) * scale))))
  y1 = max(0, min(scale, int(round(((y + h) / height) * scale))))
  return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
```

---

## 3. Token-Level BIO Alignment

For token classification in LayoutXLM:
- Tokens are assigned BIO (Beginning, Inside, Outside) tags:
  - `B-SELLER`, `I-SELLER`
  - `B-ADDRESS`, `I-ADDRESS`
  - `B-TIMESTAMP`, `I-TIMESTAMP`
  - `B-TOTAL_COST`, `I-TOTAL_COST`
  - `O` (Background / non-key tokens)

### Subword Expansion Rule
When a word is split into $K$ subwords:
- First subword receives tag: `B-{ENTITY}` (or `I-{ENTITY}` if continuing an entity from a previous word in the same span).
- Subsequent $K-1$ subwords receive tag: `I-{ENTITY}`.
- All $K$ subwords inherit the same normalized $[x_0, y_0, x_1, y_1]$ box.

---

## 4. Leakage-Free Grouped Splitting

> [!IMPORTANT]
> A simple random 80/20 train/validation split introduces massive **data leakage**: receipts from the same retail chain (e.g. VinCommerce, Co.opMart, Circle K) share identical header fonts, addresses, and layout structures.

### Splitting Protocol
1. Extract canonical merchant identifier from the first `SELLER` text in each receipt.
2. Group images by merchant identifier.
3. Apply `GroupKFold` or `StratifiedGroupKFold` to ensure all receipts from a particular merchant chain appear exclusively in *either* the training split or the validation split.
4. Validation performance then genuinely measures generalization to previously unseen invoice layouts.
