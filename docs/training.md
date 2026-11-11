# Training and evaluation guide (Week 4)

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
