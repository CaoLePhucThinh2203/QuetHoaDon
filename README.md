# QuetHoaDon - Vietnamese Receipt OCR & KIE

[![tests](https://github.com/CaoLePhucThinh2203/QuetHoaDon/actions/workflows/tests.yml/badge.svg)](https://github.com/CaoLePhucThinh2203/QuetHoaDon/actions/workflows/tests.yml)

Read a photo of a Vietnamese receipt and extract structured information (seller, address, timestamp, total) with a
**CRAFT → VietOCR → LayoutXLM** pipeline.

**Web demo:** https://quethoadon-web.vercel.app/ (upload a receipt photo; the model runs on a Hugging Face Space, so the
first request after a long pause can take a few minutes while it starts).

| Course | Introduction to Artificial Intelligence (SEB44021) |
| --- | --- |
| Lecturer | Trần Thế Vũ |
| Semester | Semester 1, academic year 2026–2027 |
| Team | Group 1 (see [Team](#team)) |

![Pipeline architecture](assets/pipeline_architecture.png)

> **Status (Week 5).** The full pipeline runs: preprocessing → CRAFT → VietOCR → LayoutXLM → field extraction. The
> Week-5 work raised reading accuracy: VietOCR was fine-tuned on MC-OCR line crops, LayoutXLM (M1b v3) was retrained on
> that OCR, and date / time and SELLER / ADDRESS clean-up rules were added. On the test split the share of receipts with
> all four fields read exactly right went from 9.6 % (Week 4) to 18.4 % (frozen configuration, scored 17/11) and 21.1 %
> (product configuration, scored 22/11), see [Results](#results-week-5-final-configuration-test-split). A rule-based
> reader for item lines (name, quantity, unit price, amount) reads the right total in 74.2 % of test receipts, scored with
> the annotated total. A web demo (`deploy/`, `notebooks/08_web_demo_kaggle.ipynb`) runs the product configuration.
> Plan and status: [`docs/reports/`](docs/reports/).

## 1. Source code

All logic lives in `src/`, one package per pipeline stage; notebooks and scripts only call it.

| Stage | Module | Status |
| --- | --- | --- |
| 0 Preprocessing | `src/preprocessing/image_preprocessor.py` | implemented, unit-tested |
| 1 Detection (CRAFT) | `src/detection/` | implemented; logic unit-tested, model call verified on Colab |
| 2 OCR (VietOCR) | `src/recognition/` | implemented; logic unit-tested, model call verified on Colab |
| 3 Features (LayoutXLM) | `src/features/` | implemented, not unit-tested end to end |
| 4 Field extraction | `src/extraction/`, `src/training/` | BIO decoding, soft decoding and keyword / date rules; M1b and M1a trained and scored (Kaggle T4) |
| 4b Item lines | `src/extraction/line_items.py` | rule based (no item labels in MC-OCR); scored by the annotated total |
| OCR fine-tuning | `src/recognition/finetune*.py` | VietOCR fine-tuned on line and CRAFT-region crops; gate on validation CER |
| Baseline B0 | `src/baseline/b0.py` | frozen features + K-means + cluster-to-field mapping; scored on validation and test |
| Orchestration | `src/pipeline/runner.py` | `ReceiptPipeline.predict()` (one image, OCR only), `.run()` (dataset, resumable) |
| Evaluation | `src/evaluation/`, `src/training/field_eval.py` | CER/WER, detection recall, field P/R/F1 (strict and OCR-tolerant), bootstrap CI, error causes |
| Shared | `src/common/` | bbox utilities, config loader |

```text
QuetHoaDon/
├── .github/workflows/tests.yml   # CI: unit tests on every push / PR
├── assets/
│   ├── pipeline_architecture.png
│   ├── preprocessing_before_after.jpg
│   └── samples/sample_receipt.jpg  # synthetic receipt for a quick test
├── configs/                      # config.yaml (paths, data policy), dataset.yaml, model.yaml
├── deploy/
│   ├── space/                    # Gradio API on a Hugging Face Space (ZeroGPU)
│   ├── api/                      # FastAPI + Docker version of the same API
│   └── web/                      # web page on Vercel and its serverless proxy to the Space
├── docs/
│   ├── architecture.md           # pipeline, data contracts, model architecture, limits
│   ├── dataset.md                # MC-OCR facts, preprocessing steps, split, data-use policy
│   ├── training.md               # how to train, score and compare the models (commands, settings)
│   ├── deploy.md                 # how to deploy the web demo (Hugging Face Space + Vercel)
│   ├── archive/                  # documents of the earlier unsupervised direction (history only)
│   └── reports/                  # project plan, status, weekly reports
├── notebooks/
│   ├── 01_mc_ocr_eda.ipynb       # dataset audit
│   ├── 02_pipeline_colab.ipynb   # full OCR run on the dataset (Colab GPU)
│   ├── 03_train_evaluate_kaggle.ipynb  # training, scoring and comparison (Kaggle T4)
│   ├── 04_week5_demo_latency_rq1_kaggle.ipynb  # end-to-end demo, latency, preprocessing ablation (Kaggle T4)
│   ├── 05_phase0_baseline_kaggle.ipynb  # accuracy baseline: strict app metrics, saved probabilities, error catalogue
│   ├── 06_phase1_ocr_kaggle.ipynb  # VietOCR fine-tuning, OCR rerun, M1b v3, scoring with date / time rules
│   ├── 07_final_evaluation_kaggle.ipynb  # frozen configuration (17/11): every reported number re-checked, demo photo -> JSON
│   ├── 08_web_demo_kaggle.ipynb  # web demo (Gradio, public link) of the product configuration
│   ├── 09_line_merge_kaggle.ipynb  # experiment: OCR of merged line segments (negative result)
│   ├── 10_digit_finetune_kaggle.ipynb  # experiment: VietOCR fine-tuning for digits (negative result)
│   └── 11_final_test_product_kaggle.ipynb  # second test run, product configuration, two self-checking gates
├── scripts/
│   ├── prepare_dataset.py        # manifests + invoice-level split
│   ├── build_token_dataset.py    # OCR JSON + annotations -> BIO token dataset
│   ├── run_pipeline.py           # one image -> OCR JSON (CLI entrypoint)
│   ├── train_layoutxlm.py        # M1b (fine-tune) and M1a (--freeze-encoder)
│   ├── run_baseline_b0.py        # baseline B0 (features, K-means, cluster-to-field)
│   ├── evaluate_layoutxlm.py     # field-level scoring of a checkpoint
│   ├── compare_models.py         # comparison table with paired bootstrap
│   ├── error_analysis.py         # error causes per field
│   ├── find_duplicates.py        # near-duplicate receipts across splits
│   ├── rescore_without_duplicates.py
│   ├── aggregate_seeds.py        # mean and standard deviation over seeds
│   ├── draw_errors.py            # draw wrong fields on the receipt for manual review
│   ├── extract_receipt.py        # end-to-end demo: photo -> four fields with M1b
│   ├── measure_latency.py        # latency and size of B0 / M1a / M1b on GPU and CPU
│   ├── ocr_preprocessing_ablation.py  # RQ1: preprocessing vs VietOCR error rate
│   ├── verify_extractor.py       # the demo path must reproduce the scored Week-4 predictions
│   ├── decode_from_probs.py      # decode saved word probabilities offline, strict app scores, error table
│   ├── score_app_metrics.py      # strict reading accuracy of saved reports
│   ├── ocr_error_catalog.py      # first sort of OCR-text errors for the manual review
│   ├── finetune_vietocr.py       # build crop / text pairs and fine-tune VietOCR (validation gate)
│   ├── run_ocr_dataset.py        # OCR of the whole dataset, optional fine-tuned weights
│   ├── filter_nested_ocr.py      # offline nested-box filter of saved OCR
│   ├── compare_ocr_runs.py       # ceiling of each OCR run on validation
│   ├── score_items.py            # item lines scored with the annotated total
│   └── compare_strict.py         # paired comparison of two decoded runs (McNemar, bootstrap)
├── src/                          # source code (table above)
├── tests/                        # 342 unit tests, no GPU needed
├── requirements.txt              # runtime (pipeline)
└── requirements-dev.txt          # light deps for the unit tests / CI
```

## 2. Documentation

- [`docs/architecture.md`](docs/architecture.md) - stages, classes and parameters, JSON data contracts,
  LayoutXLM architecture, field-extraction design, limitations.
- [`docs/dataset.md`](docs/dataset.md) - MC-OCR 2021 annotations, preprocessing steps, split figures, known
  issues, data-use policy (train / validation / test).
- [`docs/training.md`](docs/training.md) - commands and settings to train, score and compare the models.
- [`docs/deploy.md`](docs/deploy.md) - deployment of the web demo (Hugging Face Space for the model, Vercel for the page).
- [`docs/reports/`](docs/reports/) - project plan with the rubric check, weekly reports, final report
  ([`week5_final_report.md`](docs/reports/week5_final_report.md)).

## 3. Data preprocessing steps

Dataset: [MC-OCR 2021](https://www.kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021) (Kaggle,
`domixi1989/vietnamese-receipts-mc-ocr-2021`). It is not stored in the repo.

**Step 1 - clean annotations, manifests and an invoice-level 80/10/10 split** (`src/preprocessing/annotation_parser.py`,
`dataset_preparer.py`): parses the `|||`-separated texts/labels/polygons, fixes the `TOTAL_TOTAL_COST` label typo,
drops empty / invalid records and checks that every image exists.

```bash
python scripts/prepare_dataset.py --csv <dataset>/mcocr_train_df.csv \
    --images <dataset>/train_images/train_images --out outputs/dataset
```

**Step 2 - image preprocessing + OCR** (Stage 0-2, run by `notebooks/02_pipeline_colab.ipynb`): resize to 1280 px,
deskew, CLAHE + bilateral filter, then CRAFT + VietOCR; one `<id>.json` per image.

![Before / after Stage 0](assets/preprocessing_before_after.jpg)

**Step 3 - BIO token dataset for LayoutXLM** (`src/preprocessing/token_dataset.py`): splits OCR lines into words,
normalises boxes to [0, 1000], gives each word the label of the annotation polygon that contains it and writes
`train/validation/test.jsonl` plus a label-coverage report.

```bash
python scripts/build_token_dataset.py --csv <dataset>/mcocr_train_df.csv \
    --ocr-dir pipeline_results/ocr_annotations \
    --split-manifest outputs/dataset/mcocr_manifest_split.csv --out outputs/token_dataset_v1
```

Details: [`docs/dataset.md`](docs/dataset.md#preprocessing-steps).

## Results (Week 5, final configuration, test split)

Two configurations were scored on the test split (116 receipts): the **frozen configuration** on 17/11/2026 and the **product configuration** (the one the web demo runs) on 22/11/2026. The test split was therefore used **twice**; the rules of the second run were committed before it ([`docs/reports/final_test_protocol.md`](docs/reports/final_test_protocol.md)). First the frozen configuration, scored once after it was frozen. Strict reading accuracy: the predicted text
must equal the annotated text after normalisation (what a user of an upload-a-receipt app would see). "Week 4" is M1b v2
with the Week-4 decoding plus the date / time rules; "final" is fine-tuned VietOCR + nested-box filter 0.5 + M1b v3 + date /
time rules + SELLER / ADDRESS clean-up. Paired comparison over the same receipts (`scripts/compare_strict.py`, exact
McNemar test and bootstrap over receipts): SELLER +15.0 points [+7.5, +22.4] (p = 0.0004), ADDRESS +11.3 [+5.7, +17.9]
(p = 0.0005), date and time +17.0 [+9.4, +24.5] (p < 0.0001), all four fields +8.8 [+3.5, +14.0] (p = 0.002). Total
+7.1 [+1.0, +14.3] but McNemar p = 0.065, so the gain on the total is not shown to be significant.

| Exactly right | Week 4 + date / time rules | Final configuration |
| --- | --- | --- |
| SELLER | 61 / 107 (57.0 %) | 77 / 107 (72.0 %) |
| ADDRESS | 17 / 106 (16.0 %) | 29 / 106 (27.4 %) |
| Date | 83 / 106 (78.3 %) | 89 / 106 (84.0 %) |
| Time hh:mm | 43 / 85 (50.6 %) | 61 / 85 (71.8 %) |
| Total | 64 / 98 (65.3 %) | 71 / 98 (72.4 %) |
| **All four fields** | 11 / 114 (9.6 %) | 21 / 114 (18.4 %) [12.4, 26.5] |
| Lenient macro F1 | 0.664 | 0.780 |
| Fields wrong because of OCR text | 95 | 52 |

**Product configuration (22/11, same model and OCR).** Adds the item rules of 19/11, `total_from_items` (use the verified item
sum when the model's total is not printed on a total line) and `quality_check` (answer "unreadable, take another photo" when
the OCR repeats invented phrases; the 5 flagged test receipts count as not read). Paired against the frozen configuration:

| Exactly right | Frozen (17/11) | Product (22/11) | Difference [95 %] | McNemar p |
| --- | --- | --- | --- | --- |
| SELLER, ADDRESS, date, time | unchanged | unchanged | 0 | 1.0 |
| Total | 71 / 98 (72.4 %) | **80 / 98 (81.6 %)** | +9.2 [+4.1, +15.3] | **0.0039** |
| **All four fields** | 21 / 114 (18.4 %) | **24 / 114 (21.1 %)** [14.6, 29.4] | +2.6 [+0.0, +6.1] | 0.25 |
| Item sum = annotated total | 46 / 97 (47.4 %) | **72 / 97 (74.2 %)** [64.7, 81.9] | | |
| Of items reported `verified`, right | 42 / 44 (95.5 %) | 69 / 72 (95.8 %) | | |

No correct answer was lost to the "unreadable" flag (no field has a receipt that only the frozen configuration got right).
The improvements are in the total and the item lines; the other fields did not change. The same two numbers on validation
(total 81.6 %, items 74.5 %) match the test (81.6 %, 74.2 %): no sign of a validation-to-test gap for these two. The item
rules were refined once more on 23/11 (rule version 4, written on train): validation items 74.5 % -> 75.5 %; the test was
not scored again, so the test numbers above are those of version 3. Report: [`docs/reports/week5_final_report.md`](docs/reports/week5_final_report.md).

VietOCR fine-tuning (validation, region crops): CER 13.03 % -> 6.49 %. The SELLER / ADDRESS clean-up rules were written
after reading validation errors, so the validation numbers are optimistic; the test numbers above are not.

**Speed (Kaggle T4, test receipts).** OCR 2.43 s per receipt (preprocessing 0.11 s, CRAFT 0.61 s, VietOCR 1.72 s); LayoutXLM
stage 58.9 ms on GPU (p95 86.0 ms) and 913.2 ms on CPU. `scripts/verify_extractor.py` reproduces the stored Week-4
predictions of the end-to-end path on all 115 validation receipts (0 differences, PASS).

**Item lines (name, quantity, unit price, amount).** MC-OCR has no item labels, so the reader is rule based and scored by
whether the item amounts add up to the annotated total. Test, 97 receipts with an annotated total, frozen rules (17/11): at
least one item read in 80.4 %; item sum equals the annotated total in 46 / 97 (47.4 % [37.8, 57.3]); when the reader reports
a receipt as `verified` (sum equals the total printed on the receipt) 42 of 44 are right (95.5 %). With the rules of 19/11
(product configuration) the same test gives 92.8 %, 72 / 97 (74.2 %) and 69 of 72 (95.8 %). Item **names** cannot be scored
without labels. Details and limits: [`docs/reports/items_plan.md`](docs/reports/items_plan.md).

## Results (Week 4, one seed)

Field-level macro F1 on the held-out test split (116 receipts; lenient matching, 95 % bootstrap interval):

| Model | Macro F1 | Receipts with all fields right |
| --- | --- | --- |
| B0: Week 3 pipeline (frozen features + clustering + rules) | 0.352 [0.32, 0.39] | 0 / 116 |
| M1a: frozen LayoutXLM + linear head | 0.638 [0.58, 0.69] | 30 / 116 |
| M1b: fine-tuned LayoutXLM | 0.658 [0.60, 0.71] | 32 / 116 |

About 70 % of the remaining errors come from OCR text or annotations. Roughly 10 % of validation / test receipts have a
near-duplicate in train (macro F1 without them: M1b 0.642). How to train and score: [`docs/training.md`](docs/training.md).

## 4. Model development notebooks / scripts

| Notebook / script | Purpose | Where |
| --- | --- | --- |
| `notebooks/01_mc_ocr_eda.ipynb` | dataset audit: label taxonomy, missing images, resolutions, entity counts, samples | Kaggle / local |
| `notebooks/02_pipeline_colab.ipynb` | install → build pipeline from `configs/` → 5-image smoke test → full resumable run → inspect a result | Colab GPU |
| `notebooks/03_train_evaluate_kaggle.ipynb` | train M1b / M1a, run B0, score, compare, error analysis | Kaggle GPU |
| `notebooks/04_week5_demo_latency_rq1_kaggle.ipynb` | end-to-end demo, latency (RQ3), preprocessing ablation (RQ1), duplicate check (Week-4 model) | Kaggle GPU |
| `notebooks/05_phase0_baseline_kaggle.ipynb` | strict reading accuracy baseline, saved word probabilities, error catalogue | Kaggle GPU |
| `notebooks/06_phase1_ocr_kaggle.ipynb` | VietOCR fine-tuning, OCR rerun, nested-box filter, M1b v3 training and scoring | Kaggle GPU |
| `notebooks/07_final_evaluation_kaggle.ipynb` | **final configuration**: re-computes every reported number with KHỚP / LỆCH checks, latency, demo photo → JSON | Kaggle GPU |
| `notebooks/08_web_demo_kaggle.ipynb` | **web demo**: upload a receipt photo in the browser, get the fields, item lines and coloured word boxes (public Gradio link) | Kaggle GPU |
| `notebooks/09_line_merge_kaggle.ipynb`, `10_digit_finetune_kaggle.ipynb` | experiments, both negative: OCR of merged line segments; VietOCR fine-tuning for digits (`docs/reports/progress.md`) | Kaggle GPU |
| `notebooks/11_final_test_product_kaggle.ipynb` | **second test run** of the product configuration; two gates (frozen numbers reproduce, offline scoring = app output) must pass before it scores | Kaggle GPU |
| `deploy/` | web deployment: API on a Hugging Face Space and a page on Vercel (`docs/deploy.md`) | CPU |
| `scripts/run_pipeline.py` | the same pipeline on one image from the terminal | any |
| `scripts/extract_receipt.py` | receipt photo → seller, address, date and time, total, item lines (`--items --compact`) | GPU |
| `scripts/demo_app.py` | the same, as a Gradio web page (`--share` for a public link) | GPU |
| `scripts/train_layoutxlm.py`, `run_baseline_b0.py`, `evaluate_layoutxlm.py`, `compare_models.py` | train and score the three systems | GPU for training and scoring |

Models: CRAFT and VietOCR (`vgg_transformer`) are pretrained and used for inference only; LayoutXLM-base
(`microsoft/layoutxlm-base`) is fine-tuned for BIO token classification on the token dataset (M1b), or used with a
frozen encoder and a trained linear head (M1a). Commands and settings: [`docs/training.md`](docs/training.md).
Preprocessing and detection hyper-parameters are in `configs/model.yaml`.

**Evaluation (what MC-OCR can and cannot measure).** MC-OCR annotates only `SELLER`, `ADDRESS`, `TIMESTAMP`,
`TOTAL_COST`, so: field-level P/R/F1 and exact match on those four fields; detection **recall** on annotated regions
(precision is not measurable); CER/WER on matched regions; `ITEM_NAME`, `ITEM_PRICE`, `QUANTITY` only qualitatively.

## 5. Instructions for running the project

### 5.0 Installation

Python 3.11 or newer. For the unit tests only, `pip install -r requirements-dev.txt` is enough (section 5.3). For the
pipeline (a GPU is recommended):

```bash
git clone https://github.com/CaoLePhucThinh2203/QuetHoaDon.git
cd QuetHoaDon
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install craft-text-detector --no-deps
pip install vietocr --no-deps
pip install gdown einops prefetch_generator albumentations lmdb imgaug
pip install ninja
pip install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation   # LayoutXLM, about 10 min
```

`craft-text-detector` and `vietocr` pin old torch / numpy versions, hence `--no-deps`. The notebooks run these same
installs on Kaggle / Colab.

### 5.1 One image (CLI)

After the installation above:

```bash
python scripts/run_pipeline.py                                    # bundled sample, prints JSON
python scripts/run_pipeline.py --image path/to/receipt.jpg --output outputs/receipt.json --device cpu
```

Final configuration on one photo (needs the M1b v3 checkpoint and the fine-tuned VietOCR weights, see 5.4):

```bash
python scripts/extract_receipt.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
    --nested-threshold 0.5 --postprocess --items --compact --image path/to/receipt.jpg
```

Add `--total-from-items` (also on `demo_app.py`) to replace the total with the item sum when the items add up to a
printed total line and the predicted total is not on one (validation: total exactly right 76.5 % -> 81.6 %; added after
the frozen configuration, so it is off by default; see `docs/reports/items_plan.md` section 9).
Add `--quality-check` to answer "unreadable, take another photo" instead of numbers when the OCR repeats invented
phrases (`src/recognition/quality.py`; validation: 5 of 115 photos flagged, all of them with a wrong total before).

Web demo of the final configuration (same model, upload a photo in the browser). On Kaggle, run
`notebooks/08_web_demo_kaggle.ipynb` interactively (Run All, not Save & Run All) and open the printed
`https://….gradio.live` link; elsewhere, with a GPU and `pip install gradio`:

```bash
python scripts/demo_app.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth --share
```

The page hides the fields that were not read, marks item lines whose quantity × unit price differs from the amount, and
says whether the item amounts add up to the printed total. Cell 4 of notebook 08 checks that the web path returns
exactly the output of the scored path (`ReceiptExtractor.extract`) on two test receipts.

`craft-text-detector` and `vietocr` must be installed with `--no-deps` (they pin old torch/numpy versions); see the
comments in `requirements.txt`.

### 5.2 Full dataset (Colab GPU)

Open `notebooks/02_pipeline_colab.ipynb`, run the install cells (they also build detectron2 for LayoutXLM), then
the **smoke test** (5 images) before the full run. The runner resumes after a disconnect and stops early if more than
half of the first images fail. Then run Step 1 and Step 3 of [section 3](#3-data-preprocessing-steps).

### 5.3 Unit tests (no GPU)

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
```

The same command runs in CI on every push / pull request.

### 5.4 Reproducibility

The reported results were produced on a Kaggle notebook (GPU T4, Internet on) with the versions below; other versions
may need small changes, especially transformers 5.x for the LayoutXLM processor.

| Component | Version used |
| --- | --- |
| Python | 3.13 |
| torch | 2.11 (CUDA 13 build) |
| transformers | 5.20 |
| detectron2 | built from source each session, about 5 - 10 minutes |
| Seed | 42 for every reported run |

Order: run the OCR notebook, then `prepare_dataset.py` and `build_token_dataset.py` ([section 3](#3-data-preprocessing-steps)),
then the commands in [`docs/training.md`](docs/training.md) (section "Final configuration" for Week 5).
`notebooks/07_final_evaluation_kaggle.ipynb` re-computes the numbers of the **frozen configuration** from the saved models and
prints KHỚP when it matches; it is pinned to commit `413599d` because the item rules on `main` changed after it (run on `main`,
its item cell would report LỆCH). `notebooks/11_final_test_product_kaggle.ipynb` runs on `main` and re-computes both
configurations. Model weights (M1b v3 checkpoint 1.4 GB, fine-tuned VietOCR 152 MB) are not stored in the repository; the
team keeps them in a Kaggle dataset and on Google Drive, or they can be retrained with the commands in `docs/training.md`
(VietOCR about 40 minutes, OCR of the dataset about 45 minutes, M1b about 30 minutes on a T4).

## Team

| # | Full name | Student ID |
| --- | --- | --- |
| 1 | Dam Huynh Hoai Nam | 24090022 |
| 2 | Cao Le Phuc Thinh | 24090024 |
| 3 | Dau Tuan Loc | 24090025 |
| 4 | Truong Dinh Nhien | 24090060 |
