# Training and evaluation guide (Weeks 4 and 5)

Models: **B0** (Week 3 pipeline: frozen LayoutXLM features, K-means, clusters mapped to fields), **M1a** (frozen encoder +
linear head), **M1b** (LayoutXLM fully fine-tuned, `LayoutLMv2ForTokenClassification`, 9 BIO labels).

## Environment

GPU runs happen on Kaggle (T4). Pinned versions used for the reported results: Python 3.13, torch 2.11, transformers 5.20,
detectron2 built from source (`pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation`,
about 5 - 10 minutes per fresh session). Unit tests need only `requirements-dev.txt` (no GPU, no torch):
`python -m unittest discover -s tests -t .`.

Kaggle flow: dataset `quethoadon` (preprocessed images + `token_dataset_v1`) attached read-only; code comes from
`git clone` of the branch; results are written to `/kaggle/working` and must be downloaded or saved (Quick Save with
output) before the session ends.

## Data

`token_dataset_v1/{train,validation,test}.jsonl` (922 / 115 / 116 receipts) built by `scripts/build_token_dataset.py` from
the OCR output and the MC-OCR annotations. Each record: `words`, `boxes` (0 - 1000), BIO `labels`, `gt` (field texts),
`image`. Windows of at most 480 sub-tokens (overlap 100) cover receipts longer than 512 tokens; probabilities of
overlapping windows are averaged.

## Training

```bash
# M1b (fine-tune everything). v2 settings: class weights, checkpoint chosen by macro F1 over the 4 fields
python scripts/train_layoutxlm.py --token-dir $TOK --image-dir $IMG --out runs --name m1b_v2 \
    --class-weight sqrt_inverse --select-metric macro --epochs 20 --patience 5
# M1a (frozen encoder, head only)
python scripts/train_layoutxlm.py --token-dir $TOK --image-dir $IMG --out runs --name m1a \
    --freeze-encoder --lr 1e-3 --epochs 20 --patience 5 --select-metric macro
```

Common settings: AdamW (weight decay 0.01), 10 % warm-up then linear decay, batch 4 x accumulation 2, fp16, gradient
clipping 1.0, seed 42, cross-entropy over the first sub-token of each word. Each run writes `run.json` (configuration,
versions, per-epoch history, best epoch) and keeps the best checkpoint in `best/`. The test split is never read by the
training script.

| Run | LR | Class weights | Selected by | Best epoch | Val entity F1 | Time |
|---|---|---|---|---|---|---|
| M1b v1 | 5e-5 | none | micro F1 | 4 | 0.406 | 10.4 min (early stop at 7) |
| M1b v2 | 5e-5 | sqrt inverse | macro F1 | 15 | 0.428 | 29.4 min (early stop at 20) |
| M1a | 1e-3 | none | macro F1 | 16 | 0.317 | 9.5 min |

v1 overfits after epoch 3 - 4 (validation loss stops improving while training loss keeps falling). v2 changed three things
at once, so the +0.02 entity F1 cannot be attributed to class weights alone.

## Baseline B0

```bash
python scripts/run_baseline_b0.py --token-dir $TOK --image-dir $IMG --cache b0_cache --out eval/b0_val.json
```

Frozen word embeddings are cached in `--cache`. `k` is swept over 8, 12, 16, 24, 32, 48 and chosen by Silhouette on the
validation words (32; Silhouette 0.088, i.e. weak cluster structure). For the test split reuse `--k 32 --split test --allow-test`.

## Evaluation

```bash
python scripts/evaluate_layoutxlm.py --checkpoint runs/m1b_v2_*/best --token-dir $TOK --image-dir $IMG \
    --split validation --out eval/m1b_v2_val.json              # test needs --split test --allow-test
python scripts/compare_models.py --baseline B0 --report B0=eval/b0_val.json --report M1a=eval/m1a_val.json \
    --report M1b=eval/m1b_v2_val.json --out eval/comparison_val.md
python scripts/error_analysis.py --report eval/m1b_v2_val.json --out eval/error_analysis_m1b_val.md
python scripts/find_duplicates.py --token-dir $TOK --image-dir $IMG --out eval/dup_report.json
python scripts/rescore_without_duplicates.py --dups eval/dup_report.json --report M1b=eval/m1b_v2_test.json
```

Final decoding is shared by all systems (soft decoding, rejoined OCR-split amounts and dates, keyword / date rules);
see `src/training/field_eval.py`, `src/extraction/decode.py`, `src/extraction/rules.py`.

## More seeds and manual error review

```bash
python scripts/train_layoutxlm.py ... --name m1b_s1 --seed 1      # also --seed 2; same for M1a
python scripts/aggregate_seeds.py --system M1b=eval/m1b_s42_test.json,eval/m1b_s1_test.json,eval/m1b_s2_test.json
python scripts/draw_errors.py --report eval/m1b_v2_test.json --token-dir $TOK --image-dir $IMG --split test --out error_images
```

Do not choose anything with the test split; seeds only report the spread of an already frozen configuration.

## Final configuration (Week 5): fine-tuned VietOCR, M1b v3, rules, item lines

Every choice below was made on validation; the test split was scored once on 17/11/2026 with the frozen configuration.
`notebooks/06_phase1_ocr_kaggle.ipynb` holds the training run, `notebooks/07_final_evaluation_kaggle.ipynb` re-computes
every reported number with checks that print KHỚP / LỆCH.

```bash
# 1. OCR of the whole dataset with the pretrained VietOCR (reproduces token_dataset_v1)
python scripts/run_ocr_dataset.py --images $IMG_RAW --split-manifest $MANIFEST --out ocr_v1r
# 2. (crop, text) pairs from train and validation, then fine-tune VietOCR; gate = paired CER on validation region crops
python scripts/finetune_vietocr.py build --csv $CSV --ocr-dir ocr_v1r/ocr_annotations --image-dir ocr_v1r/preprocessed_images \
    --split-manifest $MANIFEST --out ocr_pairs
python scripts/finetune_vietocr.py train --pairs ocr_pairs --out vietocr_ft --epochs 12 --lr 1e-4
# 3. OCR again with the fine-tuned weights, nested-box filter 0.5, token dataset, ceiling of each OCR run
python scripts/run_ocr_dataset.py --images $IMG_RAW --split-manifest $MANIFEST --out ocr_ft --weights vietocr_ft/vietocr_finetuned.pth
python scripts/filter_nested_ocr.py --in-dir ocr_ft/ocr_annotations --out-dir ocr_ft_n05/ocr_annotations --threshold 0.5
python scripts/build_token_dataset.py --csv $CSV --ocr-dir ocr_ft_n05/ocr_annotations --split-manifest $MANIFEST --out tok_ft_n05
python scripts/compare_ocr_runs.py --reference v1=$TOK --run ft_n05=tok_ft_n05 --out eval/ocr_runs_val.json
# 4. M1b v3: same settings as M1b v2, trained on that OCR
python scripts/train_layoutxlm.py --token-dir tok_ft_n05 --image-dir ocr_v1r/preprocessed_images --out runs --name m1b_v3_ft_n05 \
    --seed 42 --class-weight sqrt_inverse --select-metric macro --epochs 20 --patience 5
# 5. score, keep probabilities, decode with the date / time rules and the SELLER / ADDRESS clean-up
python scripts/evaluate_layoutxlm.py --checkpoint runs/m1b_v3_*/best --token-dir tok_ft_n05 --image-dir $IMG \
    --split validation --save-probs eval/m1b_v3_val_probs.npz --out eval/m1b_v3_val.json
python scripts/decode_from_probs.py --token-dir tok_ft_n05 --split validation --probs eval/m1b_v3_val_probs.npz \
    --datetime --cleanup --out eval/decoded_v3_val.json
# 6. item lines (rule based) scored with the annotated total; paired comparison of two decoded runs
python scripts/score_items.py --token-dir tok_ft_n05 --split validation --out eval/items_val.json
python scripts/compare_strict.py --token-dir tok_ft_n05 --split validation --a week4=eval/decoded_val.json \
    --b final=eval/decoded_v3_val.json --out eval/paired_val.json
```

Measured on Kaggle T4: OCR of the 1153 receipts about 45 minutes; VietOCR fine-tuning about 3 minutes per epoch on
about 20 000 pairs; M1b training about 25 - 30 minutes. A second VietOCR round (start from the round-1 weights, pairs
cut from the round-1 OCR, 20 epochs at lr 5e-5) did not lower the validation CER and was dropped
(`docs/reports/items_plan.md`).

## End-to-end extraction and Week 5 measurements

```bash
# final configuration: photo -> four fields + item lines, short JSON for an app (GPU)
python scripts/extract_receipt.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
    --nested-threshold 0.5 --postprocess --items --compact --image receipt.jpg --output out.json
# Week-4 model (pretrained VietOCR), full output
python scripts/extract_receipt.py --checkpoint runs/m1b_v2_*/best --image receipt.jpg --output out.json
# the same from an existing OCR result and its preprocessed image (no CRAFT / VietOCR)
python scripts/extract_receipt.py --checkpoint runs/m1b_v2_*/best --ocr-json <id>.json --preprocessed-image <id>.jpg

# check that the demo path reproduces the scored Week-4 predictions on validation (must print PASS)
python scripts/verify_extractor.py --checkpoint runs/m1b_v2_*/best --token-dir $TOK --image-dir $IMG \
    --reference eval/m1b_v2_val.json --out eval/verify_extractor_val.json

# latency and size of the LayoutXLM stage on GPU and CPU (RQ3); --b0-model adds B0
python scripts/measure_latency.py --token-dir $TOK --image-dir $IMG --checkpoint M1b=runs/m1b_v2_*/best --out eval/latency.json

# effect of preprocessing on VietOCR, validation split only (RQ1); needs the ORIGINAL photos and the split manifest
python scripts/ocr_preprocessing_ablation.py --csv $CSV --image-dir $IMG_RAW --split-manifest outputs/dataset/mcocr_manifest_split.csv \
    --degradations none,blur,dark,noise --out eval/ocr_ablation.json --md eval/ocr_ablation.md
```

The extractor decodes with the same functions that scored the results; `--postprocess` adds the Week-5 rules and must be
used with M1b v3. A tokenizer bug made the per-receipt path (the extractor and `measure_latency.py`) cut every word into
its own window after the first processor call; it was fixed in `word_token_counts` (Week 5) and `verify_extractor.py`
printed PASS afterwards (115 / 115 receipts identical). Latency measured before that fix is wrong and was discarded. In the ablation, the four configurations are
resize only, deskew, contrast (CLAHE + bilateral filter) and both; blur, darkening and noise use fixed parameters
(`src/evaluation/ocr_ablation.py`). The ablation crops the annotated lines, while the pipeline reads CRAFT boxes, so it
measures the recogniser under each preprocessing setting rather than the end-to-end OCR error.
`notebooks/04_week5_demo_latency_rq1_kaggle.ipynb` runs all of this on Kaggle; its cell 9 also checks that the
regenerated split is the one the model was trained with.

## Accuracy baseline and offline decoding (validation only)

```bash
# score a checkpoint and keep the word probabilities
python scripts/evaluate_layoutxlm.py --checkpoint runs/m1b_v2_*/best --token-dir $TOK --image-dir $IMG --split validation \
    --save-probs eval/m1b_v2_val_probs.npz --out eval/m1b_v2_val_now.json
# decode the saved probabilities without GPU; with --reference it must reproduce that report (PASS)
python scripts/decode_from_probs.py --token-dir $TOK --split validation --probs eval/m1b_v2_val_probs.npz \
    --reference eval/m1b_v2_val_now.json --out eval/decoded_val.json
# strict reading accuracy (what an app user sees: exact amount, exact date and time, exact text) for saved reports
python scripts/score_app_metrics.py --token-dir $TOK --split validation --report M1b=eval/m1b_v2_val.json --out eval/app_metrics_val.json
# first sort of OCR-text errors for the manual review
python scripts/ocr_error_catalog.py --report eval/decoded_val.json --out-dir eval/ocr_errors
```

`notebooks/05_phase0_baseline_kaggle.ipynb` runs this sequence. Decoding or rule changes are scored on the saved
probabilities before any model is retrained; the test split needs `--allow-test` and is not part of this sequence.
