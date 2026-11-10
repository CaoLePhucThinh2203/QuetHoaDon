# Kế hoạch tổng và danh sách task

*Soạn ngày 10/11/2026, bám sát `AI Project Progress.docx` (Tuần 1–3) và yêu cầu Tuần 4–5 của giảng viên. Đây là kế hoạch hiện hành; [`project_status.md`](project_status.md) và [`week4_plan.md`](week4_plan.md) là tài liệu tham khảo, nếu mâu thuẫn thì theo file này.*

**Mục tiêu:** pipeline CRAFT → VietOCR → LayoutXLM chạy tốt nhất có thể, có một model LayoutXLM được fine-tune kỹ, vượt rõ baseline của Tuần 3, có số đo đáng tin và trả lời đủ 3 RQ.

**Ưu tiên:** P0 = bắt buộc (theo rubric hoặc docx), P1 = giúp model tốt hơn rõ rệt, P2 = tối ưu thêm hoặc phân tích sâu.

**Quy ước chạy:** test, OCR và huấn luyện chạy trên Colab; máy cá nhân chỉ dùng để sửa code và quản lý git.

## 0. Nguyên tắc bám docx

- Vấn đề, 3 RQ và dataset chính (MC-OCR 2021) giữ nguyên như Tuần 2. SROIE dùng kiểm tra sơ bộ detection và trích xuất, CORD dùng cho phân tích bố cục và mặt hàng (đúng vai trò đã nêu ở Tuần 2).
- Pipeline giữ nguyên: ảnh → tiền xử lý → CRAFT → VietOCR → LayoutXLM → trích xuất thông tin.
- **Baseline = pipeline Tuần 3** (B0): đặc trưng LayoutXLM đóng băng → phân cụm → gán cụm vào trường. **Model cải tiến = LayoutXLM được fine-tune** (M1b) để trích xuất trực tiếp, đúng ý Tuần 2 ("LayoutLM tiếp nhận chữ và toạ độ để bóc tách").
- Chia dữ liệu theo docx mục 1.3: Development (= tập train 922 hóa đơn) dùng chỉnh tham số tiền xử lý, cấu hình phân cụm và huấn luyện; Validation (115) dùng đánh giá độ ổn định và chọn tham số; Test (116) chỉ dùng một lần ở cuối.
- Bộ chỉ số theo docx mục 4 (CRAFT P/R/F1, VietOCR CER/WER, Silhouette, P/R/F1 trích xuất, Exact Match, thời gian) cộng bộ chỉ số Tuần 4 (Accuracy, F1, RMSE, AUC).

## 1. Đối chiếu rubric

| Mục | Điểm | Đề yêu cầu | Còn thiếu |
| --- | --- | --- | --- |
| 1. AI Problem Definition | 2 | Ý tưởng có ý nghĩa, định nghĩa rõ bài toán thực tế | Người dùng mục tiêu, phạm vi, tiêu chí thành công đo được |
| 2. Problem Statement và RQ | 2 | Statement súc tích, ít nhất 3 RQ dẫn hướng giải pháp deep learning | Giả thuyết, metric và thí nghiệm cho từng RQ |
| 3. Solution Development | 5 | Deep learning trả lời 3 RQ: dữ liệu, tiền xử lý, model, đánh giá, diễn giải, lý do từng quyết định | Chưa train model deep learning nào, chưa trả lời RQ nào |
| 4. GitHub | 1 | Code, docs, tiền xử lý, notebook hoặc script model, hướng dẫn chạy | Notebook chưa có output, chưa có script train và đánh giá |

## 2. Khớp với docx và yêu cầu từng tuần

### Tuần 3 (docx: Data Preparation và Modeling Approach)

| Mục docx | Nội dung docx | Task | Hiện trạng |
| --- | --- | --- | --- |
| 1.1 Cleaning | Bỏ ảnh lỗi, ảnh trùng; kiểm tra định dạng, kích thước; xử lý ảnh nhiễu, mờ; chỉnh hướng | 1.3, 1.4, 1.11 | Chỉnh hướng đã có (deskew), phần còn lại chưa |
| 1.2 Transformation | Resize giữ tỉ lệ, normalization, khử nhiễu, tăng tương phản, chỉnh hướng | Stage 0 đã có; đo hiệu quả bằng 3.2 (RQ1) | Có code, chưa có số đo |
| 1.3 Splitting | Development / Validation / Test | 1.2, 1.4 | Split 922/115/116 có, chưa xử lý ảnh trùng |
| 2 Feature engineering | 6 nhóm: chữ, toạ độ, vị trí dọc, căn hàng ngang, số, bố cục | 2.5 (chữ và toạ độ là đầu vào của LayoutXLM) | Chưa có code cho 4 nhóm sau |
| 3 Model selection | CRAFT, VietOCR, LayoutXLM và lý do | Giữ nguyên; lý do ghi trong `docs/architecture.md`; thử phương án thay thế ở giai đoạn 5 | Gần xong |
| 4 Baseline metrics | CRAFT P/R/F1 | 3.5, 3.8 | Chưa đo |
| | VietOCR CER, WER | 3.2, 3.4 | Hàm đã có, chưa chạy |
| | LayoutXLM + clustering: Silhouette | 2.2, 2.4 | Chưa |
| | Trích xuất: P/R/F1 | 2.1, 2.4 | Hàm đã có, chưa chạy |
| | Toàn pipeline: Exact Match, thời gian xử lý | 2.1, 2.3 | Thời gian từng stage đã có trong JSON |
| 4.2 Trường đích | SELLER, ADDRESS, ITEM_NAME, ITEM_PRICE, QUANTITY, TOTAL_COST (Tuần 2 còn nhắc ngày tháng) | 4 trường có nhãn (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST) đo định lượng; ITEM_NAME, ITEM_PRICE, QUANTITY đánh giá định tính ở 8.5 | Theo đúng giới hạn của dữ liệu, xem mục 3 |
| Planned tasks Tuần 4 | Kế hoạch mô hình hóa | Chính file này | Xong |

### Tuần 4 (yêu cầu: Model Development và Evaluation)

| Yêu cầu | Task | File nộp |
| --- | --- | --- |
| 1. Giải thích kiến trúc model | 4.1, 4.2, 2.4 | `docs/architecture.md` |
| 2. Quá trình train: siêu tham số, hàm loss, khó khăn | 4.2, 4.5, 0.5 | `docs/training.md`, `outputs/train/run_*.json` |
| 3. Chỉ số đánh giá: Accuracy, F1, RMSE, AUC | 2.1 | `docs/reports/week4_evaluation.md`, `outputs/metrics/*.json` |
| 4. So sánh baseline (B0) với model cải tiến (M1b) | 2.4, 4.3, 5.1 | `docs/reports/week4_evaluation.md` |
| 5. Phân tích lỗi và kế hoạch cải tiến | 8.3, 4.4–4.11 | `docs/reports/week4_error_analysis.md` |

### Tuần 5 (yêu cầu: giải pháp end-to-end)

| Yêu cầu | Task |
| --- | --- |
| Demo pipeline deep learning đầy đủ | 6.5, 9.3 |
| Tổng kết hiệu năng model cuối | 8.1, 8.2, 9.2 |
| Hạn chế và rủi ro | 9.2 |
| Hướng cải tiến | 9.2, giai đoạn 5 |
| Review GitHub cuối | 9.5 |

## 3. Ghi chú kỹ thuật (điều docx chưa lường tới, xử lý mà không đổi hướng)

| Vấn đề | Cách xử lý trong khuôn khổ docx |
| --- | --- |
| MC-OCR chỉ có nhãn 4 trường, không có nhãn món hàng | Đo định lượng 4 trường; ITEM_NAME, ITEM_PRICE, QUANTITY đánh giá định tính (8.5) |
| Precision của CRAFT không hợp lệ trên MC-OCR (chữ ngoài 4 trường không có box) | Báo recall trên MC-OCR; đo đủ P/R/F1 trên SROIE (3.8) |
| Phân cụm không tự đặt tên trường | Bước gán cụm vào trường (đa số trên Development, thêm luật từ khóa và định dạng ngày, tiền) là một phần của B0 |
| Tuần 2 ghi LayoutLM *trích xuất*, Tuần 3 ghi LayoutXLM *trích đặc trưng* | Cả hai giữ nguyên: B0 dùng LayoutXLM trích đặc trưng (Tuần 3), M1b fine-tune để trích xuất (Tuần 2 và Tuần 4) |
| LayoutXLM giới hạn 512 token | Cửa sổ trượt (4.1) |
| `label_distribution` trong docx thiếu 2 nhãn (cộng được 3033, tổng vùng 6585) | Chạy lại `prepare_dataset.py` và sửa docx (0.6) |
| Accuracy token thô cao giả vì lớp `O` áp đảo | Báo Accuracy cấp hóa đơn và F1 cấp thực thể; giải thích trong báo cáo (2.1) |

## 4. Các model so sánh

| Mã | Model | Huấn luyện | Trả lời |
| --- | --- | --- | --- |
| **B0** | **Baseline Tuần 3:** CRAFT + VietOCR + đặc trưng LayoutXLM đóng băng → phân cụm → gán cụm vào trường (đa số trên Development, thêm luật từ khóa và định dạng) | không có gradient; cấu hình chỉnh trên Development và Validation | RQ2, RQ3 |
| M1a | LayoutXLM đóng băng và head phân loại token | chỉ head | RQ2, RQ3 (ablation) |
| **M1b** | **Model cải tiến:** LayoutXLM fine-tune, BIO 9 nhãn, hậu xử lý | toàn bộ | RQ2 (model chính), RQ3 |
| OCR-ft | VietOCR fine-tune có và không augmentation | có | RQ1 |
| CAT | TF-IDF và LR so với XLM-R, phân loại chi tiêu ở cấp hóa đơn | có | RQ2 |

Nguyên tắc đánh giá: một split duy nhất (sau khi gộp ảnh trùng), cùng một bộ OCR cho mọi model, chọn cấu hình trên Validation, chạy Test đúng một lần, báo cáo kèm khoảng tin cậy bootstrap.

## 5. Danh sách task

### Giai đoạn 0. Chuẩn bị và dọn nền

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 0.1 | Bật bảo vệ nhánh `main` (PR, review, CI); đưa task list lên GitHub Issues | P0 | Board có đủ task |
| 0.2 | Đổi `data_policy` sang dùng nhãn để train; sửa README và `docs/dataset.md`; chuyển docs cũ vào `docs/archive/` | P0 | Docs không còn mâu thuẫn |
| 0.3 | Tạo Drive chung chứa OCR, token dataset, đặc trưng, checkpoint, wheel; quy ước version (`ocr_v1`, `ocr_v2`) | P0 | Cả nhóm truy cập được |
| 0.4 | Môi trường Colab: build detectron2 một lần, lưu wheel lên Drive, ghim version torch và transformers vào `requirements-colab.txt` | P0 | LayoutXLM chạy được forward trên Colab |
| 0.5 | Chuẩn hóa log thí nghiệm: ghi config, commit hash, seed, siêu tham số, loss, metrics vào `outputs/train/run_<id>.json` | P0 | Có template và hàm ghi log |
| 0.6 | Sửa docx Tuần 3: số `label_distribution`; thêm kết quả Tuần 4 và 5 | P0 | Docx khớp với repo |

### Giai đoạn 1. Dữ liệu sạch và nhãn chất lượng cao

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 1.1 | Chạy `01_mc_ocr_eda.ipynb` trên Colab, giữ output; thêm phân bố `anno_image_quality` và số vùng nhãn mỗi trường | P0 | Notebook có output |
| 1.2 | Chạy lại `prepare_dataset.py` để có đủ 4 nhãn và đường dẫn ảnh tương đối | P0 | `dataset_summary.json` đúng |
| 1.3 | Tìm khoảng 109 ảnh bị thiếu (bộ lọc `data0.7`); chọn ảnh theo `img_id` trong manifest | P0 | Có đủ 1153 ảnh |
| 1.4 | Tìm hóa đơn trùng (cùng seller, ngày, tổng tiền trong nhãn, cộng perceptual hash); gom vào cùng split rồi chốt split | P0 | Split cuối, kiểm tra leak sạch |
| 1.5 | Rà nhãn gốc: lỗi chính tả nhãn, vùng TOTAL_COST chỉ có chữ "Tổng tiền" mà thiếu số, hóa đơn thiếu trường; ghi cách xử lý (data quality review của Tuần 2) | P1 | Bảng lỗi nhãn và quyết định |
| 1.6 | Hàm chuyển polygon nhãn sang toạ độ ảnh đã xử lý (từ `scale_factor`, `skew_angle`, `processed_shape`) kèm test | P0 | Test pass trên Colab (code đã viết) |
| 1.7 | Chạy OCR cho đủ 1153 ảnh (`ocr_v1`) | P0 | OCR JSON trên Drive |
| 1.8 | `build_token_dataset.py`: gán nhãn theo `contains_ratio`, ra BIO; báo cáo độ phủ nhãn từng trường | P0 | `token_dataset_v1` và bảng độ phủ (code đã viết) |
| 1.9 | Kiểm tra tay khoảng 50 hóa đơn đã ghép nhãn (vẽ box lên ảnh); chỉnh ngưỡng ghép trên Development để độ phủ cao mà không gán sai | P1 | Độ phủ tăng, có ví dụ minh họa |
| 1.10 | Gán danh mục chi tiêu theo cửa hàng; viết guideline (siêu thị, nhà thuốc thuộc nhóm nào); gán tay cửa hàng mơ hồ | P0 | `category_labels.csv` và guideline |
| 1.11 | Làm sạch theo docx mục 1.1: bỏ ảnh không đọc được, kiểm tra định dạng và kích thước, gắn cờ ảnh mờ (phương sai Laplacian) để phân tích riêng | P0 | Danh sách ảnh bị loại hoặc gắn cờ |

### Giai đoạn 2. Khung đánh giá và baseline B0 (Tuần 3)

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 2.1 | `scripts/evaluate.py`: P/R/F1 cấp thực thể từng trường, Exact Match, Accuracy cấp hóa đơn (đúng tổng tiền), ROC-AUC từng nhãn từ xác suất token, MAE và RMSE của số tiền tổng, bootstrap CI, tách theo nhóm chất lượng ảnh | P0 | Script, test, xuất `outputs/metrics/*.json` |
| 2.2 | Trích đặc trưng LayoutXLM đóng băng (Stage 3) cho cả 1153 hóa đơn, lưu lên Drive | P0 | Đặc trưng trên Drive |
| 2.3 | Script đo latency dùng chung: thời gian từng stage trên GPU và CPU, kích thước model | P0 | Gọi được cho mọi model |
| 2.4 | B0: phân cụm đặc trưng (thử số cụm, chọn theo Silhouette trên Validation), gán cụm vào trường bằng đa số trên Development, thêm luật từ khóa và định dạng (`src/extraction/rules.py`); chấm bằng 2.1 | P0 | Silhouette và F1 của B0 trên Validation |
| 2.5 | Feature engineering theo docx mục 2: vị trí dọc tương đối, căn hàng ngang, đặc trưng số (tiền, số lượng), quan hệ khoảng cách giữa vùng chữ; dùng cho B0, hậu xử lý và ghép tên món với giá | P0 | `src/features/handcrafted.py` kèm test |

### Giai đoạn 3. Tối ưu OCR (RQ1 và nền cho trích xuất)

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 3.1 | Cắt crop các vùng có nhãn theo split, kèm text chuẩn, theo định dạng VietOCR | P0 | Dataset nhận dạng |
| 3.2 | Ablation tiền xử lý: CER và WER với 4 cấu hình (không, deskew, tăng tương phản, đầy đủ), tách theo nhóm chất lượng ảnh | P0 | Bảng và biểu đồ RQ1 |
| 3.3 | Thử ảnh làm mờ, tối, nhiễu nhân tạo, so có và không tiền xử lý | P1 | Bảng độ bền |
| 3.4 | Fine-tune VietOCR có và không augmentation; thử 2 đến 3 mức augmentation | P0 | CER và WER trên Validation của từng bản |
| 3.5 | Recall của CRAFT trên các vùng có nhãn; chỉnh ngưỡng (`text_threshold`, `low_text`, `link_threshold`) trên Validation để tăng recall | P1 | Độ phủ detection tăng |
| 3.6 | So beam search với greedy của VietOCR (độ chính xác và thời gian) | P2 | Bảng so sánh |
| 3.7 | Chọn tổ hợp OCR tốt nhất, chạy lại toàn bộ thành `ocr_v2`, tạo `token_dataset_v2` | P1 | OCR mới, độ phủ cao hơn v1 |
| 3.8 | Đo CRAFT đủ Precision, Recall, F1 trên SROIE (mọi chữ đều có box), đúng vai trò kiểm tra sơ bộ ở Tuần 2 | P1 | Bảng detection |

### Giai đoạn 4. Model cải tiến M1b: LayoutXLM fine-tune (RQ2)

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 4.1 | `src/training/`: dataset và collator (`LayoutXLMProcessor`), cửa sổ trượt cho hóa đơn dài hơn 512 token, kèm test | P0 | Module và test |
| 4.2 | Train M1b bản đầu (`LayoutLMv2ForTokenClassification`, BIO 9 nhãn, cross-entropy), learning curve, checkpoint | P0 | Số M1b v1 trên Validation |
| 4.3 | `predict.py` và hậu xử lý: ghép token thành trường, chuẩn hóa số tiền (`src/extraction/decode.py` đã viết) | P0 | JSON chung, chấm bằng 2.1 |
| 4.4 | Hậu xử lý cho TOTAL_COST: chỉ nhận đoạn đọc được thành số, dùng từ khóa gần đó, đối chiếu tiền khách đưa trừ tiền thối; khi mơ hồ trả trạng thái `uncertain` | P1 | Tổng tiền đúng tăng so với 4.3 |
| 4.5 | Tìm siêu tham số trên Validation: lr, epoch, warmup, batch, weight decay, trọng số lớp, label smoothing, max length và stride (Optuna hoặc grid nhỏ) | P1 | Bảng các lần chạy và cấu hình tốt nhất |
| 4.6 | Train trên `token_dataset_v2` và so với v1 để thấy OCR tốt hơn giúp trích xuất bao nhiêu | P1 | Bảng v1 và v2 |
| 4.7 | Augmentation cho trích xuất: dịch box nhẹ, thay chữ bằng lỗi OCR giả, đổi độ sáng ảnh | P2 | So có và không |
| 4.8 | Chạy nhiều seed (ít nhất 3) và cross-validation 5 fold trên Development và Validation để có số ổn định | P1 | Mean và độ lệch chuẩn |
| 4.9 | Ensemble: trung bình xác suất các seed hoặc fold | P2 | So với model đơn |
| 4.10 | Thử head CRF trên LayoutXLM để ràng buộc chuỗi BIO hợp lệ | P2 | So với head linear |
| 4.11 | Ngưỡng tin cậy: trường có xác suất thấp trả về `uncertain`; vẽ biểu đồ calibration | P1 | Trạng thái `found`, `uncertain`, `not_found` có nghĩa |

### Giai đoạn 5. So sánh kiến trúc

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 5.1 | M1a: LayoutXLM đóng băng, chỉ train head | P1 | So đóng băng, fine-tune và B0 |
| 5.2 | XLM-R chỉ dùng chữ (bỏ box và ảnh) để đo đóng góp của bố cục | P1 | Bảng đóng góp layout |
| 5.3 | LayoutXLM bỏ nhánh ảnh (chỉ chữ và box) | P2 | Đóng góp của ảnh |
| 5.4 | LiLT-InfoXLM làm đối chứng thêm (cũng là phương án dự phòng nếu không cài được detectron2) | P2 | Thêm một dòng trong bảng |
| 5.5 | t-SNE embedding đóng băng theo nhãn (minh họa cho B0) | P2 | Hình cho báo cáo |
| 5.6 | CORD (nếu đủ thời gian): thử trích ITEM_NAME, ITEM_PRICE, QUANTITY và phân loại mặt hàng bằng nhãn cây của CORD, đúng vai trò đã nêu ở Tuần 2 | P2 | Ví dụ và số sơ bộ |

### Giai đoạn 6. Phân loại chi tiêu (RQ2)

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 6.1 | Baseline TF-IDF và LR trên text OCR | P0 | Macro-F1 Validation |
| 6.2 | Fine-tune XLM-R (hoặc PhoBERT) phân loại hóa đơn | P0 | Macro-F1 Validation |
| 6.3 | Chia theo cửa hàng để đo khả năng gặp cửa hàng mới (tránh học thuộc tên cửa hàng) | P1 | So chia ngẫu nhiên và chia theo cửa hàng |
| 6.4 | Xử lý mất cân bằng lớp (trọng số lớp, oversampling); thêm đặc trưng tên seller lấy từ M1b | P1 | Macro-F1 tăng |
| 6.5 | Gộp vào pipeline: ảnh thành trường và danh mục | P0 | Một hàm `predict` đầu-cuối |

### Giai đoạn 7. Hiệu năng (RQ3)

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 7.1 | Đo latency và kích thước của B0, M1a, M1b, XLM-R trên GPU và CPU | P0 | Biểu đồ độ chính xác và độ trễ |
| 7.2 | Quantize int8 LayoutXLM trên CPU; thử fp16 trên GPU | P1 | Latency và F1 |
| 7.3 | Export ONNX và đo lại | P2 | Bảng so sánh |
| 7.4 | Khuyến nghị triển khai dựa trên bảng độ chính xác và độ trễ | P1 | Mục khuyến nghị trong báo cáo |

### Giai đoạn 8. Chốt và đánh giá cuối

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 8.1 | Chốt cấu hình cuối của mọi model dựa trên Validation | P0 | Danh sách cấu hình cố định |
| 8.2 | Chạy Test một lần cho tất cả model | P0 | `final_test.json` |
| 8.3 | Phân tích lỗi: ít nhất 50 hóa đơn sai, phân loại nguyên nhân (detection, OCR, ghép nhãn, cắt 512 token, model, hậu xử lý), có tỉ lệ và ví dụ | P0 | `week4_error_analysis.md` |
| 8.4 | Phân tích theo nhóm chất lượng ảnh và theo loại cửa hàng (độ thiên lệch) | P1 | Bảng và nhận xét |
| 8.5 | Đánh giá định tính ITEM_NAME, ITEM_PRICE, QUANTITY trên khoảng 20 hóa đơn vì các trường này không có nhãn | P0 | Ví dụ minh họa |

### Giai đoạn 9. Báo cáo, demo, repo

| # | Task | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 9.1 | Bộ nộp Tuần 4 theo mục 2: kiến trúc, quá trình train, chỉ số, so sánh B0 với M1b, phân tích lỗi và kế hoạch cải tiến; cập nhật docx và slide | P0 | Đủ 5 mục Tuần 4 |
| 9.2 | Báo cáo cuối: bài toán và tiêu chí thành công, RQ và giả thuyết, dữ liệu, phương pháp kèm lý do, kết quả theo RQ, phân tích lỗi, hạn chế và rủi ro, hướng phát triển, tài liệu tham khảo | P0 | Báo cáo hoàn chỉnh |
| 9.3 | Demo Gradio trên Colab (ảnh thành JSON, danh mục, thời gian, độ tin cậy) và video dự phòng | P0 | Demo chạy được |
| 9.4 | Đưa checkpoint tốt nhất lên Hugging Face Hub hoặc Drive, có model card | P1 | Link trong README |
| 9.5 | Repo cuối: README (kết quả chính và cách chạy), notebook có output, docs cập nhật, CI xanh, không commit dữ liệu | P0 | Rubric mục 4 đủ |
| 9.6 | Slide Tuần 5 và tổng duyệt; mỗi người giải thích được phần mình làm | P0 | Hoàn tất |
| 9.7 | Ký trang cam kết chống đạo văn, gửi email nộp bài cho giảng viên | P0 | Đã nộp |

## 6. Thứ tự tối ưu model

1. Dữ liệu sạch và ghép nhãn tốt (giai đoạn 1).
2. Baseline B0 của Tuần 3 và khung đánh giá (giai đoạn 2).
3. Tối ưu OCR (giai đoạn 3).
4. M1b: bản đầu, hậu xử lý, siêu tham số, chuyển sang OCR v2, nhiều seed và ensemble (giai đoạn 4).
5. Các so sánh và phân tích còn lại (giai đoạn 5 đến 8).

Ở mỗi vòng chỉ giữ thay đổi nào làm tăng số trên Validation. Test chỉ chạy một lần ở cuối.

## 7. Rủi ro chính

| Rủi ro | Cách xử lý |
| --- | --- |
| Không build được detectron2 | Cache wheel trên Drive, ghim version; cuối cùng chuyển sang LiLT-InfoXLM |
| Độ phủ nhãn thấp, nhất là TOTAL_COST | Báo cáo con số này trước khi train; chỉnh ngưỡng ghép trên Development |
| M1b không vượt B0 | Vẫn là kết quả hợp lệ: phân tích nguyên nhân (nhãn, OCR, dữ liệu ít) và nêu hướng cải tiến |
| Ảnh trùng giữa các split làm số đo ảo | Làm task 1.4 trước khi train |
| Hết GPU Colab | Lưu checkpoint mỗi epoch lên Drive; dự phòng Kaggle |
