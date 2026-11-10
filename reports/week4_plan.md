# Kế hoạch Tuần 4 — Model Development & Evaluation

> **Cập nhật 10/11/2026:** nhóm chọn **phương án B** (fine-tune LayoutXLM, có giám sát) và bỏ notebook DesignThinking.
> Baseline B0 chính là pipeline Tuần 3 (đặc trưng LayoutXLM đóng băng, phân cụm, gán cụm vào trường kèm luật). M1 dưới đây
> gọi là M1b; M1a (LayoutXLM đóng băng, chỉ train head) là thí nghiệm đối chứng. Danh sách task đầy đủ và bảng khớp với
> docx: [`project_plan.md`](project_plan.md); nếu mâu thuẫn thì theo file đó.

**Đầu ra Tuần 4:** *một model đã huấn luyện, đã đánh giá, và lộ trình cải tiến rõ ràng.*

## 0. Quyết định cần chốt ngay ngày 1 (30 phút họp)

Yêu cầu Tuần 4 gồm *training process (hyperparameters, loss), so sánh baseline với improved model*. Pipeline hiện
tại toàn bộ là model pretrained dùng để suy luận — **không có bước huấn luyện, không có loss**. Nhóm phải chọn một:

| | **Phương án B (khuyến nghị)** | Phương án A |
| --- | --- | --- |
| Baseline | CRAFT + VietOCR + **trích xuất bằng luật** (không huấn luyện) | như B |
| Improved | **Fine-tune LayoutXLM phân loại token** (BIO) trên split train | clustering LayoutXLM + luật tốt hơn |
| "Training" và loss | có: cross-entropy, siêu tham số, learning curve | không có loss; "siêu tham số" = ngưỡng/luật/cụm |
| Chính sách dữ liệu | nhãn MC-OCR được dùng để train → **phải sửa README/config, không gọi là unsupervised** | giữ nguyên "unsupervised" |
| Vì sao | MC-OCR có nhãn 4 trường; repo đã chia train/val/test; plan Tuần 2 vốn dùng LayoutLM để trích xuất; đáp ứng đủ tiêu chí Tuần 4 | ít việc hơn nhưng khó chứng minh có "model đã huấn luyện" |

Phần còn lại của kế hoạch viết theo **phương án B**; nếu chọn A, bỏ WS-C và thay bằng cải tiến luật + phân tích cụm.
Sau khi chốt: cập nhật `configs/config.yaml` (`data_policy`), `README.md`, `docs/dataset.md`.

## 1. Đầu ra bắt buộc và nơi lưu

| # | Yêu cầu Tuần 4 | File nộp |
| --- | --- | --- |
| 1 | Giải thích kiến trúc model | `docs/architecture.md` §4-5 (cập nhật head, số nhãn, sơ đồ) |
| 2 | Quá trình huấn luyện: hyperparameters, loss, khó khăn | `docs/training.md` + `outputs/train/*.json` (config, learning curve) |
| 3 | Chỉ số đánh giá | `reports/week4_evaluation.md` + `outputs/metrics/*.json` |
| 4 | So sánh baseline và improved | bảng trong `reports/week4_evaluation.md` |
| 5 | Phân tích lỗi + kế hoạch cải tiến | `reports/week4_error_analysis.md` |

## 2. Cái gì đo được trên MC-OCR

- Trường có nhãn: `SELLER`, `ADDRESS`, `TIMESTAMP`, `TOTAL_COST`. `ITEM_*`, `QUANTITY`: chỉ đánh giá định tính trên ~20 hóa đơn mẫu.
- Detection: chỉ tính được **recall** trên vùng có nhãn; **precision không hợp lệ** (chữ không có nhãn bị tính là dương tính giả).
- CER/WER: chỉ trên vùng khớp với vùng có nhãn (IoU ≥ 0.5).
- Báo cáo trên **validation** để chọn cấu hình, trên **test** đúng một lần cuối.

### Chỉ số

| Chỉ số | Áp dụng cho | Ghi chú |
| --- | --- | --- |
| Precision / Recall / F1 (cấp thực thể, `seqeval`) | trích xuất 4 trường | chỉ số chính |
| Exact match theo hóa đơn | `TOTAL_COST` (so số đã chuẩn hóa), các trường còn lại (so chuỗi chuẩn hóa) | `src/evaluation` |
| Recall @ IoU 0.5 | CRAFT | dùng `xywh_to_xyxy` cho nhãn MC-OCR |
| CER, WER | VietOCR | `corpus_error_rate`, không lấy trung bình của tỉ lệ |
| Thời gian / ảnh theo từng stage | toàn pipeline (RQ3) | đã có `timing_sec` trong JSON |
| Silhouette | B0 (LayoutXLM đóng băng + phân cụm) | chỉ số baseline theo docx Tuần 3 |
| Accuracy | cấp hóa đơn: đúng tổng tiền, khớp toàn hóa đơn | không dùng ở cấp token vì nhãn `O` chiếm đa số nên accuracy cao nhưng vô nghĩa |
| RMSE, MAE | sai số số tiền tổng (VND) | yêu cầu Tuần 4; báo cả MAE vì lỗi OCR tạo ngoại lệ lớn |
| ROC-AUC | từng nhãn, từ xác suất token | yêu cầu Tuần 4; nêu rõ AUC đo khả năng xếp hạng của token chứ không đo chất lượng thực thể |

## 3. Thiết kế thí nghiệm

| Mã | Mô hình | Huấn luyện | Dùng để |
| --- | --- | --- | --- |
| **B0** | Pipeline Tuần 3: đặc trưng LayoutXLM đóng băng, phân cụm, gán cụm vào trường + luật | không có gradient | baseline |
| **M1** | Pipeline + LayoutXLM + head phân loại token, fine-tune | train split | improved |
| Ablation RQ1 | B0 với `enhance` bật/tắt; deskew bật/tắt | không | trả lời RQ1 |
| Latency RQ3 | B0 và M1 | — | trả lời RQ3 |

**Luật trong B0 (bước gán ngữ nghĩa sau phân cụm, tối thiểu):** `TOTAL_COST` = số tiền gần từ khóa "tổng/thanh toán/total" ở nửa dưới hóa đơn, trả cả ứng viên thay thế;
`TIMESTAMP` = regex ngày/giờ; `SELLER` = dòng chữ nổi bật ở đầu; `ADDRESS` = dòng giống địa chỉ ngay dưới seller.

**Gán nhãn cho M1:** chạy pipeline trên mọi ảnh → với mỗi từ OCR, gán nhãn của vùng nhãn MC-OCR có IoU/độ phủ lớn nhất
(≥ ngưỡng đã chốt), còn lại `O`. Tỉ lệ vùng nhãn không khớp được vùng OCR nào là **trần recall** của M1.

### Cấu hình huấn luyện M1 (đề xuất khởi điểm, chưa kiểm chứng)

| Mục | Giá trị khởi điểm |
| --- | --- |
| Nhãn | BIO cho 4 trường → 9 nhãn |
| Loss | cross-entropy, bỏ qua `-100` (token phụ, padding); thử thêm trọng số lớp vì `O` chiếm đa số |
| Optimizer | AdamW, weight decay 0.01, warmup 10 %, giảm tuyến tính |
| Learning rate | thử {2e-5, 5e-5} |
| Batch | 4 (+ tích lũy gradient 2) với độ dài 512; chỉnh theo bộ nhớ T4 |
| Epoch | tối đa 15, dừng sớm theo F1 validation |
| Khác | seed 42, fp16 autocast nếu backbone ảnh cho phép, lưu checkpoint lên Drive |

**Khó khăn dự kiến (ghi vào `docs/training.md` kèm cách xử lý):** nhãn gán theo IoU bị nhiễu; mất cân bằng lớp;
hóa đơn dài > 512 token; lỗi OCR lan sang nhãn; chỉ ~922 hóa đơn train; giới hạn GPU/thời gian của Colab;
cài detectron2 mất vài phút mỗi phiên.

## 4. Phân công theo nhóm việc (điền tên người phụ trách)

### WS-A — Dữ liệu & nhãn · Người phụ trách: ______

1. Chạy `scripts/prepare_dataset.py` trên Colab, lưu `dataset_summary.json` với **đủ 4 nhãn**.
2. Đối chiếu **1044 ảnh vs 1153 hóa đơn**; liệt kê ảnh thiếu; sửa bộ chọn ảnh.
3. Kiểm tra ảnh gần trùng giữa các split (perceptual hash); nếu có, gộp vào cùng split và chạy lại.
4. Sửa `image_path` trong manifest thành đường dẫn tương đối.
5. Viết `scripts/build_token_dataset.py`: ghép OCR JSON với nhãn → file token/BIO cho train/validation/test.
6. Chạy và lưu output cho `notebooks/01_mc_ocr_eda.ipynb`.

**Xong khi:** có 3 file split, báo cáo tỉ lệ vùng nhãn khớp được OCR, kiểm tra leak sạch.

### WS-B — Baseline & đánh giá · Người phụ trách: ______

1. Chạy smoke test (5 ảnh) rồi chạy toàn bộ validation + test; xác nhận hai bản sửa (`inhomogeneous shape`, `drop_nested`).
2. Viết `scripts/evaluate_pipeline.py`: recall detection, CER/WER, thời gian/ảnh bằng `src/evaluation`.
3. Cài đặt luật B0 trong `src/extraction/rules.py` + test; tính P/R/F1 và exact match theo trường.
4. Chạy ablation `enhance` / deskew (RQ1) trên validation.

**Xong khi:** `outputs/metrics/baseline_{val,test}.json` và bảng số liệu B0 cho 4 trường.

### WS-C — Huấn luyện M1 · Người phụ trách: ______

1. `src/training/`: dataset, `LayoutXLMForTokenClassification`, vòng huấn luyện, đánh giá `seqeval`.
2. `notebooks/03_train_layoutxlm.ipynb`: chạy trên Colab, lưu checkpoint + learning curve.
3. Thử 2 learning rate và có/không trọng số lớp; chọn theo F1 validation.
4. Đánh giá M1 trên test đúng một lần.

**Xong khi:** có checkpoint, `outputs/train/run_*.json` (siêu tham số, loss theo epoch) và `outputs/metrics/m1_{val,test}.json`.

### WS-D — Phân tích & báo cáo · Người phụ trách: ______

1. Cập nhật `docs/architecture.md` (kiến trúc M1) và viết `docs/training.md`.
2. Viết `reports/week4_evaluation.md`: bảng B0 vs M1 + trả lời RQ1/RQ3.
3. Phân tích lỗi: lấy ≥ 50 hóa đơn sai, phân loại nguyên nhân (detection bỏ sót / OCR sai / gán nhãn sai / cắt 512 token / luật sai), đếm tỉ lệ từng loại, kèm ví dụ.
4. Viết kế hoạch cải tiến xếp theo mức tác động, nêu rõ việc nào chuyển sang Tuần 5.
5. Cập nhật slide trình bày Tuần 4.

**Xong khi:** 5 file ở mục 1 đầy đủ và số liệu khớp với `outputs/metrics`.

## 5. Lịch gợi ý (7 ngày)

| Ngày | Việc | Phụ thuộc |
| --- | --- | --- |
| 1 | Họp chốt mục 0; hợp nhất repo mới; WS-A.1-2, WS-B.1 (smoke test) | — |
| 2 | WS-B chạy full val/test; WS-A.3-4 | smoke test đạt |
| 3 | WS-B.2-3 (metrics + luật B0); WS-A.5 (token dataset) | ngày 2 |
| 4 | WS-C.1-2 (code + chạy thử trên tập nhỏ); WS-B.4 (ablation) | WS-A.5 |
| 5 | WS-C.3 (chạy chính + chọn cấu hình); WS-D.1 | WS-C.2 |
| 6 | WS-C.4 (test một lần); WS-D.2-3 | WS-C.3 |
| 7 | WS-D.4-5; rà soát repo, README, slide | ngày 6 |

## 6. Rủi ro

| Rủi ro | Cách giảm |
| --- | --- |
| Pipeline vẫn lỗi khi chạy full | Smoke test + tự dừng sớm; xem `failed.json` |
| Trần recall thấp do OCR/detection | Báo cáo trần này trước khi huấn luyện; chỉnh `low_text`, `text_threshold` trên validation |
| Ảnh gần trùng giữa split làm số đo ảo | WS-A.3 trước khi huấn luyện |
| M1 không vượt B0 | Vẫn nộp: nêu số liệu thật và phân tích nguyên nhân (đây là kết quả hợp lệ) |
| Hết GPU Colab | Lưu checkpoint lên Drive mỗi epoch; runner có resume |
| Chạm test nhiều lần | Chọn cấu hình bằng validation; test chạy một lần |

## 7. Danh sách kiểm tra cuối tuần

- [ ] Quyết định mục 0 đã ghi vào README và `configs/config.yaml`
- [ ] Số liệu đủ 4 nhãn trong `dataset_summary.json`; 1044 vs 1153 đã giải thích
- [ ] Số liệu B0 và M1 trên validation và test, có chỉ số theo từng trường
- [ ] Learning curve, siêu tham số, seed, checkpoint
- [ ] Bảng so sánh + phân tích lỗi có ví dụ và tỉ lệ từng loại
- [ ] `python -m unittest discover -s tests -t .` xanh; CI xanh
- [ ] Không commit dữ liệu, checkpoint, `pipeline_results/`
