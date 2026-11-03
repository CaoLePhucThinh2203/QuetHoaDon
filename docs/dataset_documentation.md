# Dataset Documentation: MC-OCR 2021

## 1. Overview & Source

The dataset used in this project is based on the **MC-OCR 2021 Challenge** (Mobile-Captured Image Document Recognition for Vietnamese Receipts), organized in conjunction with the **IEEE RIVF 2021** International Conference on Communications & Computing.

| Attribute | Verified Value |
| :--- | :--- |
| **Dataset Title** | Vietnamese Receipts MC_OCR 2021 |
| **Kaggle Identifier** | `domixi1989/vietnamese-receipts-mc-ocr-2021` |
| **Primary Tasks** | Task 1: Image Quality Assessment (IQA); Task 2: Key Information Extraction (KIE) |
| **Target Language** | Vietnamese (full diacritics, Latin script) |
| **Image Modality** | Mobile photos under real-world distortions (perspective skew, shadows, crumples, noise) |

---

## 2. Kaggle Access & Loading Instructions

### In Kaggle Notebooks (Recommended)
1. Open or create a Kaggle Notebook.
2. In the right panel, select **Add Input** -> Search: `vietnamese-receipts-mc-ocr-2021`.
3. The dataset mounts read-only at:
   ```text
   /kaggle/input/vietnamese-receipts-mc-ocr-2021/
   ```

### Via Kaggle CLI (Local Sample Inspection Only)
> [!WARNING]
> Do not check downloaded dataset files into Git. The `.gitignore` is pre-configured to reject all files inside `data/`.
```bash
kaggle datasets download -d domixi1989/vietnamese-receipts-mc-ocr-2021 -p data/raw --unzip
```

---

## 3. Directory Layout & File Verification

From physical inspection of the unpacked dataset archive, the file structure is arranged as follows:

```text
/kaggle/input/vietnamese-receipts-mc-ocr-2021/
├── mcocr_train_df.csv                  # Main ground-truth annotations (1,155 records)
├── mcocr_val_sample_df.csv             # Public validation submission template (391 records)
├── train_images/
│   └── train_images/                   # Nested directory containing 1,155 .jpg image files
│       ├── mcocr_public_145013aagqw.jpg
│       └── ...
├── val_images/
│   └── val_images/                     # Nested directory containing 391 .jpg image files
│       ├── mcocr_val_145114xmicn.jpg
│       └── ...
├── text_recognition_train_data.txt     # Line-level OCR training pairs
├── text_recognition_val_data.txt       # Line-level OCR validation pairs
└── results.csv                         # Baseline submission sample
```

> [!NOTE]
> The image directories contain a double nesting (`train_images/train_images/` and `val_images/val_images/`). Code must resolve paths dynamically rather than assuming a single-level directory.

---

## 4. Annotation Schema (Verified vs Inferred)

### Verified Fields in `mcocr_train_df.csv`

| Column Name | Data Type | Verified Format / Content |
| :--- | :--- | :--- |
| `img_id` | `str` | Filename matching image in `train_images/train_images/` (e.g. `mcocr_public_145013ddcph.jpg`) |
| `anno_polygons` | `str` (evaluable as list of dicts) | List of COCO-like dictionaries per annotated line: `{'category_id': int, 'segmentation': [[x1, y1, ...]], 'area': int, 'bbox': [x, y, w, h], 'width': int, 'height': int}` |
| `anno_texts` | `str` | Text transcriptions separated by delimiter `\|\|\|` (e.g. `MINIMART ANAN\|\|\|Chợ Sủi Gia Lâm`) |
| `anno_labels` | `str` | Entity category tags separated by `\|\|\|` (e.g. `SELLER\|\|\|ADDRESS\|\|\|TOTAL_COST`) |
| `anno_num` | `int` | Count of annotated entity bounding boxes in the image |
| `anno_image_quality` | `float` | Quality score between `0.0` and `1.0` representing readability |

### Category ID to Label Mapping (Verified)
- `15` -> `SELLER`
- `16` -> `ADDRESS`
- `17` -> `TIMESTAMP`
- `18` -> `TOTAL_COST`

### Verified Data Anomalies
1. **Empty Records:** Exactly 2 rows (`img_id: mcocr_public_145014iwhec.jpg` and `mcocr_public_145014jndnz.jpg`) have `anno_num: 0`, `anno_polygons: '[]'`, and empty strings for `anno_texts` and `anno_labels`.
2. **Label Typos:** In row 8 (`mcocr_public_145013kzjew.jpg`), the label string contains `TOTAL_TOTAL_COST`. The parser must normalize this to `TOTAL_COST`.
3. **Compound Total Annotations:** Both the key header (e.g., `"Tổng tiền:"`) and the numerical figure (e.g., `"74,000"`) are assigned the label `TOTAL_COST`.

---

## 5. Properties Requiring Further Exploration

1. **Non-Key Text Presence:** The verified annotations in `mcocr_train_df.csv` cover primarily key target entities. Non-key background text (itemized products, unit prices, tax codes) does not have explicit ground truth bounding boxes in `mcocr_train_df.csv`. To construct a complete document layout for LayoutXLM, either CRAFT must detect all words first, or LayoutXLM must be trained on key lines + inferred background tokens.
2. **Merchant Distribution:** How many unique merchant chains (e.g. VinCommerce, Co.opMart, Circle K) exist across the 1,155 images? This dictates how group splitting must be executed to prevent data leakage.
3. **Image Dimensions & Aspect Ratios:** Distribution of widths, heights, and orientations across the 1,155 images.

---

## 6. Licensing & Usage Restrictions

- **Source:** Released under academic research competition terms for IEEE RIVF 2021.
- **Permitted Use:** Academic study, university research, educational benchmarking, and non-commercial prototyping.
- **Redistribution Policy:** Raw data cannot be republished commercially or packaged into closed-source proprietary software.
- **Credentials & API Safety:** No Kaggle credentials, API tokens, or session cookies may be stored or committed to the repository.
