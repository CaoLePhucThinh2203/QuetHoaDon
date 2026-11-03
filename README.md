# QuetHoaDon

A research prototype for understanding Vietnamese receipts and extracting structured information from receipt images. The project explores a pretrained computer-vision and document-understanding pipeline using CRAFT, VietOCR, and LayoutXLM, followed by unsupervised grouping and rule-based candidate validation.

> **Project status:** The repository currently provides the project structure, configuration, documentation, annotation parser, utility functions, schemas, and unit tests. It does not yet contain a verified end-to-end inference pipeline. The models and processing stages described below are the intended architecture, not a claim that the complete system is already operational.

---

## 📌 Project Overview

Vietnamese receipts are often captured by mobile phones and may contain skew, perspective distortion, uneven lighting, faded text, and complex layouts. These conditions make it difficult to identify and associate information such as store names, item descriptions, prices, and the final amount.

**QuetHoaDon** aims to process a receipt image and extract:

- **`SELLER`**: Store or company name.
- **`ADDRESS`**: Merchant address or location.
- **`ITEM_NAME`**: Purchased item description.
- **`ITEM_PRICE`**: Price associated with each item.
- **`TOTAL_COST`**: Final receipt total.
- **`QUANTITY`** _(optional)_: Item quantity, when it can be identified reliably.
- **`TIMESTAMP`** _(optional)_: Transaction date and time, when available.

The output is designed to preserve the original text, bounding boxes, confidence information when available, and uncertainty status. Ambiguous values should be marked for review instead of being silently treated as correct.

---

## 🏗️ System Architecture

The proposed processing pipeline contains five main stages:

```mermaid
flowchart LR
    A["Receipt Image"] --> B["Image Preprocessing"]
    B --> C["Stage 1: CRAFT Text Detection"]
    C --> D["Polygon Rectification and Cropping"]
    D --> E["Stage 2: VietOCR Text Recognition"]
    E --> F["OCR Text + Bounding Boxes"]
    F --> G["Stage 3: LayoutXLM Feature Extraction"]
    G --> H["Unsupervised Grouping and Clustering"]
    H --> I["Semantic Candidate Assignment"]
    I --> J["Spatial, Textual and Arithmetic Validation"]
    J --> K["Structured JSON + Uncertainty Status"]
```

### Stage 0 — Image Preprocessing

Prepare the input receipt image for text detection. Depending on image quality, preprocessing may include resizing, color conversion, denoising, and deskewing. Original image dimensions and coordinate information should be retained so that detected regions can be mapped back to the source image.

### Stage 1 — CRAFT Text Detection

CRAFT is used as a pretrained text detector to locate text regions in the receipt image. It returns polygons or bounding boxes around text.

The detected coordinates and detection scores should be preserved for later processing.

### Stage 2 — VietOCR Text Recognition

Detected text regions are cropped and rectified where necessary. The pretrained VietOCR model then converts each crop into Vietnamese text while preserving Vietnamese diacritics.

Each recognized text region should remain associated with its original bounding box and detection information. Detection confidence and OCR confidence are separate signals.

### Stage 3 — LayoutXLM Feature Extraction

LayoutXLM receives recognized text together with normalized 2D bounding boxes. Its pretrained hidden representations are used as contextual and layout-aware features.

In this project, LayoutXLM is intended to act as a **feature extractor**, not as a ready-made entity classifier. The project does not currently define a supervised BIO token-classification head trained on receipt entity labels.

### Stage 4 — Unsupervised Grouping and Information Extraction

Text regions are grouped using multiple signals:

- LayoutXLM contextual representations.
- Normalized bounding-box coordinates and spatial relationships.
- Text patterns, such as alphabetic, numeric, and currency-like content.
- Alignment of text regions into rows and columns.
- Relationships between nearby text regions.

The system then identifies possible seller names, addresses, item descriptions, item prices, and total amounts. Arithmetic and layout checks can help validate item-price associations and total candidates.

Clustering by itself does not assign meanings such as `SELLER` or `TOTAL_COST`. A separate candidate-interpretation step is required. If multiple candidates conflict or the evidence is weak, the result should retain an uncertainty status.

---

## 🧠 Unsupervised Learning Approach

The project follows an **inference-first, unsupervised extraction strategy**.

This means that receipt-specific ground-truth entity labels are not used to train or fine-tune a task-specific extraction classifier.

The project uses pretrained components:

- CRAFT for text detection.
- VietOCR for text recognition.
- LayoutXLM for contextual and layout-aware feature extraction.

These pretrained models may have been trained using supervised data by their original developers. Using their pretrained weights for inference or feature extraction does not mean that the project trains its own receipt extraction model with labeled receipt entities.

**Important:** LayoutXLM embeddings and clustering do not automatically identify the semantic meaning of a text region. Semantic candidate assignment must be implemented as a separate step using spatial information, text patterns, structural relationships, and consistency checks.

---

## 📊 Dataset

### MC-OCR 2021 — Primary Dataset

The project currently identifies **MC-OCR 2021** as its primary Vietnamese receipt dataset.

- **Dataset:** MC-OCR 2021.
- **Language:** Vietnamese.
- **Purpose:** Dataset inspection, annotation auditing, and final evaluation.
- **Kaggle source:** [`domixi1989/vietnamese-receipts-mc-ocr-2021`](https://www.kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021).

The dataset contains receipt images and annotations. In accordance with the project's unsupervised approach, entity labels are reserved for dataset auditing and final evaluation. They must not be used to train the extraction model or to tune the inference pipeline while claiming an unsupervised experiment.

The dataset's availability, exact file counts, and split sizes should be verified in the environment where the experiment is run. No dataset statistics or model performance are claimed by this README.

### Planned Dataset Integrations

| Dataset     | Language   | Status          |
| ----------- | ---------- | --------------- |
| MC-OCR 2021 | Vietnamese | Primary dataset |
| SROIE 2019  | English    | Planned         |
| CORD        | Indonesian | Planned         |

SROIE and CORD are future integration candidates. Their availability or evaluation is not implied by their presence in the configuration. Different languages may require suitable OCR recognition models.

For more information, see [Dataset Documentation](docs/dataset_documentation.md) and [Multi-Dataset Architecture](docs/multi_dataset_architecture.md).

---

## 📂 Repository Structure

```text
QuetHoaDon/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── config.yaml              # Runtime paths and data policy
│   ├── dataset.yaml             # Dataset settings and extraction fields
│   └── model.yaml               # Pretrained model and extraction settings
│
├── docs/
│   ├── architecture.md          # System architecture and processing stages
│   ├── dataset_documentation.md # Dataset usage and annotation policy
│   ├── multi_dataset_architecture.md
│   └── preprocessing.md         # Preprocessing and coordinate handling
│
├── notebooks/
│   └── kaggle/
│       └── 01_mc_ocr_eda.ipynb  # MC-OCR exploratory data analysis
│
├── reports/
│   ├── project_leader_handover.md
│   └── week3_progress_report.md
│
├── src/
│   ├── common/
│   │   └── bbox_utils.py        # Bounding-box and geometry utilities
│   ├── detection/
│   │   └── detector.py          # Text detector interface
│   ├── extraction/
│   │   └── schema.py            # Structured extraction schema
│   ├── preprocessing/
│   │   └── annotation_parser.py # MC-OCR annotation parser
│   └── recognition/
│       └── recognizer.py        # Text recognizer interface
│
└── tests/
    ├── test_annotation_parser.py
    ├── test_bbox_utils.py
    └── test_schema.py
```

---

## 🚀 Quickstart & Setup

### 1. Local Environment Setup (Lightweight)

Clone the repository and install core dependencies:

```bash
git clone https://github.com/CaoLePhucThinh2203/QuetHoaDon.git
cd QuetHoaDon

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Unit Tests

Run the full automated test suite:

```bash
python3 -m unittest discover -s tests -v
```

### 3. Kaggle EDA Workflow

Dataset exploration and data-quality checks are performed on Kaggle:

1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Create a new notebook and click **Add Input** in the right-hand panel.
3. Search and attach the dataset: **`domixi1989/vietnamese-receipts-mc-ocr-2021`**.
4. Import and execute `notebooks/kaggle/01_mc_ocr_eda.ipynb`.
5. The notebook automatically detects dataset paths, validates annotations, displays sample visualizations, and exports `mc_ocr_data_quality_report.json` to `/kaggle/working/`.

---

## 🧪 Evaluation Plan

After the complete pipeline is implemented, evaluate its stages separately:

| Component            | Suggested evaluation                                 |
| -------------------- | ---------------------------------------------------- |
| Text detection       | Precision, recall, and IoU-based matching            |
| Text recognition     | Character Error Rate (CER) and Word Error Rate (WER) |
| Field extraction     | Precision, recall, and F1-score for each field       |
| Item extraction      | Item-name and item-price association accuracy        |
| Total extraction     | Exact match or normalized amount accuracy            |
| End-to-end system    | Receipt-level extraction accuracy                    |
| Uncertainty handling | Coverage and uncertain-result rate                   |

MC-OCR annotations should only be used for final evaluation after the inference pipeline and its thresholds have been fixed. Evaluation results should be reported separately and should not be presented as training results.

---

## ⚠️ Current Implementation Status

The current repository contains:

- Bounding-box conversion, normalization, denormalization, and IoU utility functions.
- Abstract interfaces for text detection and text recognition.
- An MC-OCR annotation parser for dataset auditing and evaluation.
- A structured extraction schema.
- YAML configuration files for datasets, runtime settings, and model components.
- Unit tests for geometry utilities, annotation parsing, and schema behavior.
- Architecture, dataset, preprocessing, and project progress documentation.

The following parts are **not yet verified as implemented end to end**:

- Concrete CRAFT model loading and inference.
- Polygon perspective rectification and text-crop generation.
- Concrete VietOCR model loading and inference.
- LayoutXLM tokenizer and encoder inference.
- Unsupervised feature construction and clustering.
- Semantic candidate assignment for receipt fields.
- Item-name and item-price association.
- Arithmetic validation and uncertainty resolution.
- End-to-end JSON export and evaluation.

The README describes the intended system design. Update this section as implementation and testing progress.

---

## 📚 Documentation

- [System Architecture](docs/architecture.md)
- [Dataset Documentation](docs/dataset_documentation.md)
- [Multi-Dataset Architecture](docs/multi_dataset_architecture.md)
- [Preprocessing](docs/preprocessing.md)
- [Week 3 Progress Report](reports/week3_progress_report.md)
- [Project Leader Handover](reports/project_leader_handover.md)

---

## 👥 Team

Add the project members, roles, and responsibilities here when the team assignments are finalized.

---

## 📄 License

Add the project's license and dataset-use notices here before public distribution. The dataset and pretrained models may have their own terms of use; check the corresponding sources before redistributing data or model files.
