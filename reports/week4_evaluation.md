# Week 4 evaluation: B0 vs M1a vs M1b

Source of every number: `scripts/evaluate_layoutxlm.py`, `scripts/run_baseline_b0.py`, `scripts/compare_models.py`,
`scripts/error_analysis.py` (reports are produced on Kaggle; see [`docs/training.md`](../docs/training.md)).
Code version used for scoring: commit `5b8d708` on branch `HoaiNam/m1b-training` (checkpoints: M1b v2 trained at `51f6259`,
M1a at `8cbfb5a`). One training seed (42); 115 validation and 116 test receipts.

## Protocol

* One split (922 / 115 / 116 receipts, seed 42). Configuration chosen on validation; test scored after freezing it.
* After the first test scoring the gold-total parser was fixed (a region such as `27,000 27,000` was read as one
  absurd number). Predictions were unchanged; all three systems were re-scored with the same rule. Nothing else was
  tuned on test.
* Same OCR output and the same final decoding for every system: soft decoding for TOTAL_COST / TIMESTAMP, OCR-split
  amounts and dates rejoined (`107,0` + `,000`, `14/08/2` + `2020`), keyword rule for an empty total, first date-shaped
  word for TIMESTAMP (chosen on validation), timestamps without a date dropped.
* Metrics: lenient field F1 (SELLER / ADDRESS CER <= 0.3, TIMESTAMP same calendar date, TOTAL_COST same amount, a gold
  region may hold several numbers), receipt exact match, 95 % bootstrap intervals over receipts, paired bootstrap
  between systems. Strict string equality is also stored in the reports (`fields`).

## Comparison (soft_rules, split = validation)

| System | SELLER F1 | ADDRESS F1 | TIMESTAMP F1 | TOTAL_COST F1 | Macro F1 | Receipt all-correct | Total exact / with gold | Token AUC | ms / receipt |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.000 [0.000, 0.000] | 0.108 [0.054, 0.165] | 0.732 [0.647, 0.806] | 0.519 [0.430, 0.607] | 0.340 [0.303, 0.375] | 0.009 [0.000, 0.026] | 55 / 98 | 0.815 | - |
| M1a | 0.654 [0.560, 0.734] | 0.459 [0.369, 0.555] | 0.753 [0.674, 0.825] | 0.648 [0.558, 0.732] | 0.628 [0.568, 0.688] | 0.278 [0.191, 0.357] | 67 / 98 | 0.877 | 33 |
| M1b | 0.688 [0.603, 0.764] | 0.500 [0.407, 0.597] | 0.759 [0.678, 0.831] | 0.689 [0.605, 0.769] | 0.659 [0.602, 0.716] | 0.296 [0.209, 0.374] | 71 / 98 | 0.829 | 33 |

Ceiling with this OCR (gold tags decoded): SELLER 0.792, ADDRESS 0.643, TIMESTAMP 0.657, TOTAL_COST 0.753

### Paired bootstrap against B0 (difference, 95 % interval, share of resamples better)

| System | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | Macro F1 | Receipt all-correct |
|---|---|---|---|---|---|---|
| M1a | +0.654 [+0.562, +0.736] (1.00) | +0.351 [+0.264, +0.447] (1.00) | +0.020 [+0.000, +0.047] (0.95) | +0.129 [+0.045, +0.216] (1.00) | +0.289 [+0.243, +0.335] (1.00) | +0.270 [+0.183, +0.357] (1.00) |
| M1b | +0.688 [+0.601, +0.766] (1.00) | +0.392 [+0.301, +0.491] (1.00) | +0.027 [+0.006, +0.057] (0.98) | +0.170 [+0.095, +0.251] (1.00) | +0.319 [+0.278, +0.363] (1.00) | +0.287 [+0.208, +0.374] (1.00) |

## Comparison (soft_rules, split = test)

| System | SELLER F1 | ADDRESS F1 | TIMESTAMP F1 | TOTAL_COST F1 | Macro F1 | Receipt all-correct | Total exact / with gold | Token AUC | ms / receipt |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.000 [0.000, 0.000] | 0.065 [0.020, 0.114] | 0.788 [0.711, 0.849] | 0.553 [0.465, 0.651] | 0.352 [0.317, 0.386] | 0.000 [0.000, 0.000] | 60 / 98 | 0.839 | - |
| M1a | 0.685 [0.603, 0.765] | 0.469 [0.369, 0.565] | 0.794 [0.716, 0.857] | 0.604 [0.505, 0.697] | 0.638 [0.582, 0.691] | 0.259 [0.190, 0.345] | 64 / 98 | 0.894 | 32 |
| M1b | 0.700 [0.614, 0.779] | 0.502 [0.400, 0.597] | 0.794 [0.716, 0.857] | 0.634 [0.545, 0.716] | 0.658 [0.600, 0.709] | 0.276 [0.198, 0.362] | 64 / 98 | 0.841 | 33 |

Ceiling with this OCR (gold tags decoded): SELLER 0.864, ADDRESS 0.616, TIMESTAMP 0.727, TOTAL_COST 0.786

### Paired bootstrap against B0 (difference, 95 % interval, share of resamples better)

| System | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | Macro F1 | Receipt all-correct |
|---|---|---|---|---|---|---|
| M1a | +0.685 [+0.602, +0.764] (1.00) | +0.404 [+0.309, +0.499] (1.00) | +0.006 [+0.000, +0.019] (0.64) | +0.051 [-0.021, +0.125] (0.92) | +0.287 [+0.249, +0.322] (1.00) | +0.259 [+0.181, +0.336] (1.00) |
| M1b | +0.700 [+0.615, +0.778] (1.00) | +0.437 [+0.343, +0.533] (1.00) | +0.006 [+0.000, +0.019] (0.64) | +0.081 [+0.005, +0.167] (0.98) | +0.306 [+0.264, +0.346] (1.00) | +0.276 [+0.198, +0.353] (1.00) |

## Near-duplicate receipts across splits

`scripts/find_duplicates.py`: 109 flagged pairs, 24 across splits (21 same date + total + seller, 9 same OCR word set,
8 near-identical image hash). 12 / 115 validation and 11 / 116 test receipts have a near-duplicate in another split.
Macro F1 on the remaining receipts (`scripts/rescore_without_duplicates.py`):

| System | Test, all | Test, without duplicates (105) | Validation, all | Validation, without duplicates (103) |
|---|---|---|---|---|
| B0 | 0.352 | 0.338 | 0.340 | 0.332 |
| M1a | 0.638 | 0.620 | 0.628 | 0.623 |
| M1b | 0.658 | 0.642 | 0.659 | 0.655 |

The ranking is unchanged; the leak inflates the test score by about 0.015 - 0.02 macro F1. The "same labels" signal can
flag two different receipts of one shop (same date and amount), so this is an upper bound of the leak.

## Error analysis (M1b)

Every wrong field is assigned one cause by rule (an estimate, not a manual review; `scripts/draw_errors.py` draws the
receipts for a manual check).

## Error analysis (test, m1b_v2_test.json)

Receipts: 116, with at least one wrong field: 84. Wrong fields: 159.

| Cause | Meaning | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | All | Share |
|---|---|---|---|---|---|---|---|
| annotation_without_amount | Annotation of the total holds no number | 0 | 0 | 0 | 10 | 10 | 6% |
| no_ocr_word_labelled | No OCR word carries the field (text detection or label mapping missed it) | 1 | 1 | 0 | 0 | 2 | 1% |
| ocr_text_error | OCR text of the field is wrong, even a perfect tagger fails | 14 | 40 | 30 | 15 | 99 | 62% |
| model_missed | OCR allowed the answer, the model predicted nothing | 0 | 0 | 0 | 7 | 7 | 4% |
| model_wrong | OCR allowed the answer, the model predicted a wrong value | 16 | 12 | 1 | 12 | 41 | 26% |


Validation: Receipts: 115, with at least one wrong field: 81. Wrong fields: 160.

First rows of the test error table (full 100 rows in `error_analysis_m1b_test.md` of the evaluation zip):


| # | Receipt | Field | Cause | Gold | Predicted | Best possible from OCR |
|---|---|---|---|---|---|---|
| 1 | mcocr_public_145013amzul | TOTAL_COST | model_wrong | 112,000 / Tổng cộng | 2,000 | 2000 |
| 2 | mcocr_public_145013byufi | ADDRESS | ocr_text_error | 17A, KP.11,P.BTD, Q. Bình Tân | 1 1 BTD Bình Tân Điện | KP 1 1 BTD Bình Tân |
| 3 | mcocr_public_145013byufi | TIMESTAMP | ocr_text_error | Ngày 22/01/2020 Giờ: 16:49 | - | Ngày: 22/01 2020 Giờ: 16 49 |
| 4 | mcocr_public_145013byufi | TOTAL_COST | annotation_without_amount | Tổng thành toàn | 1,360 | 1,360,000 |
| 5 | mcocr_public_145013chgcz | SELLER | model_wrong | VinCommerce | TP | Vincommerce |
| 6 | mcocr_public_145013chgcz | ADDRESS | ocr_text_error | VM + QNH 86 Trần Phú / Số 86 Đường Trần Phú, P.Cẩm Tây / TP  | VMt - QNH 8 1861 6 Trần P Phú Số 8 Ố 85 B 6 Đường 1 g Trần í | VMt - QNH 8 1861 6 Trần P Phú Số 8 Ố 85 B 6 Đường 1 g Trần í |
| 7 | mcocr_public_145013chgcz | TOTAL_COST | model_missed | -164.435 / TỔNG TIỀN PHẢI T.TOÁN | - | - |
| 8 | mcocr_public_145013cleap | ADDRESS | ocr_text_error | VM+ QNH 112 Thanh Niên / Số 112 Thanh Niên, P. Cẩm Thành / T | VM4 - ONH 112 2 Thanh Hi Niên 8112 Thanh Hi Niên P. Cam Thàn | VM4 - ONH 112 2 Thanh Hi Niên 8112 Thanh Hi Niên P. Cam Thàn |
| 9 | mcocr_public_145013cleap | TIMESTAMP | ocr_text_error | Ngày bán: 11/08/2020 18:02 | 1/08/2 2020 | Ngày bán: 1/08/2 2020 18 3:02 |
| 10 | mcocr_public_145013cleap | TOTAL_COST | ocr_text_error | TỔNG TIỀN PHẢI T.TOÁN / 222.000 | 22,000 | 22,000 |
| 11 | mcocr_public_145013crqqo | ADDRESS | ocr_text_error | 186 Cây Trâm, P.11, Q.Gò Vấp, TP.HCM | Vietnam 1 9 www.1 Gò Vấp TP. HCM 186 Cây Trâm, P 11 Q TOÁN H | Gò Vấp TP. HCM 186 Cây Trâm, P 11 Q |
| 12 | mcocr_public_145013crqqo | TIMESTAMP | ocr_text_error | 02.10.2020 15.30 | - | 02 10 2020 15 30 |
| 13 | mcocr_public_145013cxtop | SELLER | model_wrong | LOTTERIA | LOTTERIA i'm | LOTTERIA |
| 14 | mcocr_public_145013cxtop | TIMESTAMP | ocr_text_error | 15:51 / 09/08/2020 | - | 09 3/08 3/2020 15:51 |
| 15 | mcocr_public_145013dhzmu | TOTAL_COST | annotation_without_amount | Tổng thanh toán | 4,750,466 | - |
| 16 | mcocr_public_145013dxavh | ADDRESS | ocr_text_error | VM+ QNH 112 Thanh Niên / Số 112 Thanh Niên, P. Cẩm Thành / T | VM4 - QNH 112 La Thanh H Niên Số 112 Thanh H Lạ Niên, biên,  | VM4 - QNH 112 La Thanh H Niên Số 112 Thanh H Lạ Niên, biên,  |
| 17 | mcocr_public_145013dxavh | TIMESTAMP | ocr_text_error | Thời gian: 22:02 / Ngày: 11/08/2020 | - | Ngày: 11 08/2020 Thời gian: 22:02 |

## Known limits

* One seed (`scripts/aggregate_seeds.py` is ready for more); intervals reflect receipt sampling only.
* TIMESTAMP is decided by a rule shared by all systems, so it says little about the models.
* M1b - M1a is small (about +0.02 - 0.03 macro F1); the bootstrap interval barely excludes 0.
* About 70 % of the remaining errors are on the OCR / annotation side; improving OCR (RQ1) is the main lever.

