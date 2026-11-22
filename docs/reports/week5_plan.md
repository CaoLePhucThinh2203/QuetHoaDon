# Kế hoạch Tuần 5 — nâng độ chính xác tới mức cao nhất có thể

*Viết lại ngày 17/11/2026. Thay bản cùng tên soạn sáng 17/11 (bản cũ còn trong lịch sử Git). Nhóm quyết định ưu tiên **độ chính
xác** hơn hạn chót: model phải đọc hóa đơn chuẩn nhất có thể, vì sau môn học có thể đưa vào một ứng dụng tải hóa đơn lên.
Các yêu cầu Tuần 5 của giảng viên vẫn giữ (mục 8).*

## 0. Trạng thái (cập nhật 22/11/2026)

| Giai đoạn | Kết quả |
| --- | --- |
| 0. Mốc chuẩn | Xong (notebook 05): chỉ số đọc chuẩn, xác suất lưu sẵn, tái lập PASS |
| 1. OCR | 1.2–1.3 xong: VietOCR tinh chỉnh, CER vùng validation 13,03 % → 6,49 %; bộ lọc hộp chồng 0,5; vòng 2 không cải thiện (kết quả âm). 1.1, 1.4–1.6 chưa làm |
| 2. Model | 2.1 xong (M1b v3 trên OCR tinh chỉnh); 2.3 xong, kết quả âm (3 seed và ensemble không hơn); 2.2, 2.4, 2.5 không làm |
| 3. Hậu xử lý | 3.2, 3.3 xong (luật ngày giờ, làm sạch SELLER / ADDRESS); 3.1 thử "tiền khách đưa − tiền thối", kết quả âm nên bỏ; 3.4 cố ý không làm (học thuộc tên cửa hàng của train) |
| 4. Tập ảnh thật R | Không làm: nhóm không gán nhãn tay được; ghi vào hạn chế |
| 5. Chốt và bàn giao | 5.1 PASS, 5.2 test chấm một lần (17/11), 5.3 độ trễ đo lại sau khi sửa lỗi, 5.7 tài liệu học; 5.4 và 5.6 qua notebook 07; 5.5 báo cáo cuối `week5_final_report.md` |
| Mở rộng mặt hàng | Xong bằng luật, chấm bằng tổng tiền chuẩn (`items_plan.md`); luật bản 3 (19/11): test 47,4 % → 74,2 % |
| Sau 18/11 (`progress.md`) | Luật mặt hàng v2–v3; tổng tiền từ các món (test 71 → 80 / 98); chặn ảnh khó đọc (5 / 116 test, không mất kết quả đúng nào); OCR đọc theo đoạn (âm) và tinh chỉnh chữ số (âm) |
| Cấu hình sản phẩm và chấm test lần hai (22/11) | Cùng model, thêm luật mới; quy tắc viết trước (`final_test_protocol.md`), notebook 11 với hai cổng; cả bốn trường 18,4 % → 21,1 % (p = 0,25), tổng tiền p = 0,0039 |
| Web | API trên Hugging Face Space + trang Vercel (`deploy/`, `docs/deploy.md`), Gradio trên Kaggle (notebook 08); chạy cấu hình sản phẩm |

Quyết định ở mục 9: (1) test chấm lại một lần cho cấu hình cuối và ghi rõ (đã làm ngày 22/11 cho cấu hình sản phẩm); (2) không làm tập R; (3) làm xong trước khi nộp.

## 1. Mốc xuất phát (đã kiểm chứng)

Nguồn: `week4_eval.zip` → `eval/m1b_v2_val.json`, `eval/m1b_v2_test.json` (model M1b v2, run `m1b_v2_20261110_183118`).

| Trường | F1 chịu lỗi OCR (val) | F1 khớp nghiêm ngặt (val) | Trần với OCR hiện tại (val) | F1 chịu lỗi OCR (test) |
| --- | --- | --- | --- | --- |
| SELLER | 0.688 | 0.484 | 0.792 | 0.700 |
| ADDRESS | 0.500 | 0.118 | 0.643 | 0.502 |
| TIMESTAMP | 0.759 | 0.019 | 0.657 | 0.794 |
| TOTAL_COST | 0.689 | 0.626 | 0.753 | 0.634 |

- Đúng cả 4 trường (chịu lỗi OCR): validation 34 / 115, test 32 / 116.
- Tổng tiền đúng: validation 71 / 98, test 64 / 98.
- "Trần" là kết quả khi giải mã từ **nhãn chuẩn** trên chữ OCR hiện tại: model gán nhãn hoàn hảo cũng không vượt được mức này.
  TIMESTAMP vượt trần vì luật ngày đọc được ngày ở những chỗ nhãn token không có.
- "Khớp nghiêm ngặt" là so chuỗi sau chuẩn hóa. Đây là mức người dùng ứng dụng thấy; mức chịu lỗi OCR chấp nhận sai tới 30% ký tự.

## 2. Lỗi nằm ở đâu (validation, 160 trường sai của M1b)

| Trường | OCR đọc sai chữ | Model chọn sai | Nhãn tổng tiền không có số |
| --- | --- | --- | --- |
| SELLER | 22 | 10 | 0 |
| ADDRESS | 39 | 18 | 0 |
| TIMESTAMP | 37 | 0 | 0 |
| TOTAL_COST | 19 | 7 | 8 |
| **Cộng** | **117 (73%)** | **35 (22%)** | **8 (5%)** |

Tổng tiền trên 98 hóa đơn validation có số tiền chuẩn: model xếp đúng số lên đầu 63, số đúng có trong OCR nhưng xác suất
quá thấp 15, **số đúng không có trong kết quả OCR 20**.

Kết luận dẫn hướng: đòn bẩy lớn nhất là **OCR** (nâng trần), sau đó mới tới model gán nhãn và hậu xử lý.

## 3. Chỉ số "đọc chuẩn" dùng từ nay

Ngoài các chỉ số Tuần 4 (giữ để so sánh), thêm bộ chỉ số theo góc nhìn người dùng ứng dụng, tính trên validation cho **mọi**
cấu hình, kể cả mốc M1b v2:

| Trường | Đúng khi |
| --- | --- |
| TOTAL_COST | Số tiền bằng số tiền chuẩn |
| TIMESTAMP | Ngày bằng ngày chuẩn; báo riêng tỉ lệ đúng cả giờ:phút |
| SELLER, ADDRESS | Chuỗi bằng chuỗi chuẩn sau chuẩn hóa (chữ thường, khoảng trắng, bỏ tiền tố "Địa chỉ:", "ĐC:"); báo thêm CER trung bình |
| Cả hóa đơn | Mọi trường có nhãn đều đúng theo các tiêu chí trên |

## 4. Nguyên tắc

1. Mọi lựa chọn (cấu hình OCR, siêu tham số, luật hậu xử lý) chỉ dựa trên **validation**. Chỉ giữ thay đổi làm tăng điểm
   validation và có khoảng tin cậy bootstrap ủng hộ; thay đổi nhỏ trong mức nhiễu thì không giữ.
2. Tập **test MC-OCR đã chấm một lần ở Tuần 4.** Cần nhóm chốt (mục 9): chấm lại test đúng một lần cho cấu hình cuối và ghi rõ
   trong báo cáo, hoặc chỉ báo validation.
3. Thêm **tập ảnh thật R** do nhóm tự chụp và gán nhãn, chưa từng dùng để chọn gì: đây là con số trung thực nhất cho ứng dụng.
4. Không ghi số nào vào báo cáo nếu không có file kết quả. Mọi lần chạy model trên Kaggle; máy cá nhân chỉ sửa code và Git.
5. Trước mọi thí nghiệm, `verify_extractor.py` phải in PASS (code hiện tại cho lại đúng dự đoán Tuần 4).

## 5. Danh sách task

**P0** bắt buộc; **P1** nên làm; **P2** chỉ làm nếu các bước trước đã có lợi ích rõ. "Code" là phần mình viết, "Kaggle" là phần bạn chạy.

### Giai đoạn 0. Mốc chuẩn và chẩn đoán

| # | Việc | Ưu tiên | Ai | Xong khi |
| --- | --- | --- | --- | --- |
| 0.1 | Notebook 04 cell 1–4b: unit test và kiểm chứng tái lập | P0 | Kaggle | PASS |
| 0.2 | Chỉ số "đọc chuẩn" (mục 3) cho mọi báo cáo, kèm chuẩn hóa ngày giờ | P0 | Code | Có số mốc M1b v2 trên validation |
| 0.3 | Lưu xác suất từng từ khi chấm (`--save-probs`) để thử hậu xử lý không cần GPU | P0 | Code | File xác suất của validation |
| 0.4 | Vẽ 30 hóa đơn validation sai và xem tay, phân loại lỗi OCR: dấu tiếng Việt, chữ số, chữ bị tách hoặc dính, dòng bị cắt, dòng bị bỏ sót | P0 | Kaggle + nhóm | Bảng loại lỗi OCR có số đếm |

Giai đoạn 0 đã có code (0.2 đến 0.4): `src/evaluation/app_metrics.py`, `probs_io.py`, `offline_decode.py`, `ocr_errors.py`,
các script `score_app_metrics.py`, `decode_from_probs.py`, `ocr_error_catalog.py`, cờ `--save-probs` của
`evaluate_layoutxlm.py`, và notebook `notebooks/05_phase0_baseline_kaggle.ipynb` (cell 3 và 5 là hai cổng kiểm chứng:
đúng checkpoint, tái lập PASS). Xem tay: mở ảnh trong `error_images_val` và điền cột `manual_type` của tệp
`ocr_errors_to_review.csv`; phân loại tự động chỉ là gợi ý.

### Giai đoạn 1. OCR (nâng trần)

| # | Việc | Ưu tiên | Ai | Xong khi |
| --- | --- | --- | --- | --- |
| 1.1 | Thí nghiệm tiền xử lý (RQ1, notebook 04 cell 9–11): chọn cấu hình theo CER validation | P0 | Kaggle | Bảng CER/WER, cấu hình được chọn |
| 1.2 | Thử nghiệm VietOCR tự huấn luyện: kiểm tra bộ huấn luyện của VietOCR có chạy được trên Kaggle (Python 3.13, torch 2.11) không. Nếu không, ghi lý do và bỏ 1.3 | P0 | Code + Kaggle | Chạy được 100 bước, hoặc có ghi chú lý do |
| 1.3 | Fine-tune VietOCR trên các dòng có nhãn của tập train, có và không augmentation (phần augmentation của RQ1) | P1 | Code + Kaggle | CER validation của từng bản |
| 1.4 | Gộp box CRAFT thành dòng trước khi đọc (VietOCR được huấn luyện để đọc nguyên dòng), vẫn giữ toạ độ từng từ cho LayoutXLM | P1 | Code + Kaggle | CER và trần trên validation, có và không gộp dòng |
| 1.5 | Quét ngưỡng CRAFT (`text_threshold`, `low_text`, `link_threshold`) theo độ phủ vùng nhãn | P1 | Code + Kaggle | Bộ ngưỡng chọn trên validation |
| 1.6 | Beam search so với greedy của VietOCR | P2 | Kaggle | Bảng CER và thời gian |
| 1.7 | Chạy OCR toàn bộ 1153 ảnh với tổ hợp tốt nhất → `ocr_v2`, `token_dataset_v2`; độ phủ nhãn không được giảm so với v1 (≥ 0.994) | P0 nếu 1.1–1.5 có cải thiện | Kaggle | Báo cáo độ phủ và trần mới |

### Giai đoạn 2. Model gán nhãn

| # | Việc | Ưu tiên | Ai | Xong khi |
| --- | --- | --- | --- | --- |
| 2.1 | Train lại M1b trên `token_dataset_v2`, cùng cấu hình v2 | P0 nếu có v2 | Kaggle | Điểm validation so với mốc |
| 2.2 | Quét nhỏ siêu tham số: lr {3e-5, 5e-5}, trọng số lớp {không, căn bậc hai}; chọn theo macro F1 validation | P1 | Kaggle | Bảng các lần chạy |
| 2.3 | Ba seed và ensemble trung bình xác suất | P1 | Code + Kaggle | Trung bình ± độ lệch; điểm ensemble |
| 2.4 | Augmentation lỗi OCR: train trên cả hai bản OCR (v1 và v2) của tập train | P2 | Code + Kaggle | So có và không |
| 2.5 | Lớp CRF hoặc focal loss | P2 | Code + Kaggle | Chỉ làm nếu 2.1–2.3 còn lỗi "model chọn sai" đáng kể |

### Giai đoạn 3. Hậu xử lý (không cần GPU khi đã có xác suất từ 0.3)

| # | Việc | Ưu tiên | Ai | Xong khi |
| --- | --- | --- | --- | --- |
| 3.1 | Tổng tiền: ưu tiên từ khóa "thanh toán", kiểm tra chéo "tiền khách đưa − tiền thối", loại số ngoài khoảng hợp lý | P0 | Code | Tổng tiền đúng trên validation, có và không luật mới |
| 3.2 | Ngày giờ: trích và chuẩn hóa `dd/mm/yyyy hh:mm` | P0 | Code | Tỉ lệ đúng ngày và đúng giờ |
| 3.3 | Địa chỉ, người bán: bỏ tiền tố nhãn, nối dòng đúng thứ tự | P0 | Code | Khớp nghiêm ngặt tăng hoặc giữ nguyên |
| 3.4 | Khớp tên người bán với danh sách người bán của tập train khi đủ giống (ngưỡng chọn trên validation). Báo cả có và không, vì chỉ giúp với cửa hàng đã gặp | P1 | Code | Hai con số |

### Giai đoạn 4. Tập ảnh thật R (con số cho ứng dụng)

| # | Việc | Ưu tiên | Ai | Xong khi |
| --- | --- | --- | --- | --- |
| 4.1 | Mỗi thành viên chụp khoảng 25 hóa đơn (nhiều loại cửa hàng, điện thoại, ánh sáng khác nhau) | P0 | Nhóm | Khoảng 100 ảnh |
| 4.2 | Gán nhãn theo mẫu CSV (tệp ảnh, người bán, địa chỉ, ngày, giờ, tổng tiền) | P0 | Nhóm | File nhãn |
| 4.3 | Script chấm trên tập R bằng bộ chỉ số mục 3, chạy **một lần** với cấu hình cuối | P0 | Code + Kaggle | Bảng điểm tập R |

### Giai đoạn 5. Chốt, đánh giá cuối và bàn giao

| # | Việc | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| 5.1 | Chốt cấu hình trên validation; chạy `verify_extractor.py` với cấu hình cuối | P0 | PASS với checkpoint mới |
| 5.2 | Chấm test MC-OCR theo quyết định ở mục 9 | P0 | File kết quả |
| 5.3 | Đo độ trễ cấu hình cuối, GPU và CPU (RQ3) | P0 | `latency.json` |
| 5.4 | Cập nhật demo (notebook 04) dùng cấu hình cuối | P0 | Notebook có output |
| 5.5 | Báo cáo cuối `week5_final_report.md`, README, `docs/architecture.md`, `docs/dataset.md` | P0 | Mọi số truy được về file |
| 5.6 | Review GitHub cuối: CI xanh, notebook có output, phiên bản, cách tái lập, cách lấy trọng số | P0 | Repo sạch |
| 5.7 | Chia phần học và báo cáo cho 3 thành viên | P0 | Tài liệu hướng dẫn học |

## 6. Thứ tự và điểm dừng

1. Giai đoạn 0 (mốc chuẩn) → giai đoạn 1 (OCR) → 1.7 chỉ chạy khi 1.1–1.5 cải thiện trên validation.
2. Giai đoạn 2 trên OCR tốt nhất; giai đoạn 3 chạy song song vì chỉ cần file xác suất.
3. Giai đoạn 4 (nhóm chụp và gán nhãn) bắt đầu ngay, song song với mọi thứ.
4. Giai đoạn 5 sau cùng.

Điểm dừng: một bước không cải thiện validation thì giữ cấu hình cũ, ghi kết quả âm tính vào báo cáo và chuyển bước sau.

## 7. Rủi ro

| Rủi ro | Xử lý |
| --- | --- |
| Bộ huấn luyện của VietOCR không chạy trên môi trường Kaggle hiện tại | Task 1.2 kiểm tra trước; không chạy được thì ghi vào hạn chế |
| Chọn quá nhiều thứ trên 115 hóa đơn validation dẫn tới quá khớp validation | Chỉ giữ cải thiện có khoảng tin cậy ủng hộ; tập R là phép kiểm độc lập |
| Hết GPU Kaggle (30 giờ mỗi tuần) | Ưu tiên P0; mỗi lần chạy OCR toàn bộ khoảng 45 phút, train M1b khoảng 30 phút (đã đo); các việc khác chưa đo thời gian |
| Gán nhãn tập R tốn công | Bắt đầu sớm; 100 ảnh là mức tối thiểu đề xuất, ít hơn thì khoảng tin cậy rộng hơn |
| Ảnh thật khác MC-OCR (quán nhỏ, viết tay) | Báo riêng điểm theo loại hóa đơn trong tập R |

## 8. Yêu cầu Tuần 5 của giảng viên

| Yêu cầu | Task |
| --- | --- |
| Demo pipeline Data → Model → Output | 5.4 (đã có `extract_receipt.py`, notebook 04) |
| Tổng kết hiệu năng cuối | 5.1–5.3, 4.3 |
| Hạn chế và rủi ro | 5.5 |
| Hướng cải tiến | 5.5 |
| Review GitHub (code, tài liệu, báo cáo, tái lập) | 5.6 |

## 9. Quyết định cần nhóm chốt

1. Test MC-OCR: chấm lại một lần cho cấu hình cuối và ghi rõ, hay chỉ báo validation và tập R?
2. Nhóm có chụp và gán nhãn tập R khoảng 100 hóa đơn không?
3. Hạn chót mới cho bản cuối, để xếp lượng việc giai đoạn 1–3 cho vừa.
