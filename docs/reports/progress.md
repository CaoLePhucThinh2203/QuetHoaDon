# Tiến độ sau báo cáo cuối (18–22/11/2026)

**Trạng thái 22/11/2026.** Công việc 18–19/11 (nhánh `thinh-dev`, đã merge) cải thiện tổng tiền và mặt hàng; cấu hình đó là
**cấu hình sản phẩm** (bản chốt đang chạy trên web). Ngày 22/11 nó được chấm trên tập test một lần theo quy tắc viết trước
([`final_test_protocol.md`](final_test_protocol.md), mục 7): tổng tiền đúng 71 → 80 / 98 (p = 0,0039), cả bốn trường 18,4 % →
21,1 % (p = 0,25), tổng các món bằng tổng chuẩn 47,4 % → 74,2 %, 5 ảnh khó đọc bị chặn mà không mất kết quả đúng nào. Báo cáo
cuối: [`week5_final_report.md`](week5_final_report.md). Phần dưới là nhật ký 18–19/11 (số validation, luật viết trên train).

Nhánh `thinh-dev` (gộp mọi việc từ 18/11). Ưu tiên: **tổng tiền, tên hàng, đơn giá, số lượng** (người bán, địa chỉ, ngày giờ
không cải thiện thêm). Luật viết trên **train**, validation chấm một lần mỗi vòng, test **không chấm lại trong giai đoạn này**
(lần chấm test thứ hai là ngày 22/11, xem trên). Chi tiết và file nguồn: [`items_plan.md`](items_plan.md) mục 7–10.

## Đã làm

| Vòng | Sửa gì | File |
| --- | --- | --- |
| Mặt hàng bản 2 | Gom dòng theo độ nghiêng của ảnh; gộp số OCR tách đôi (`399` + `9,000`); giữ tên hàng dính vào dòng "Giá đăng ký" | `src/extraction/line_items.py` |
| Mặt hàng bản 3 | Hàng cân theo kg (`40 000/KG X 0,322 KG`: số lượng 0,322, không còn đọc cân nặng thành 322 đồng); nhận tiêu đề cột bị OCR đọc sai; bỏ qua dòng "Tổng" giả ở đầu hóa đơn; "giò heo" không bị bỏ | `src/extraction/line_items.py` |
| Số lượng thập phân | `0, 52 290,000 150,800` và `0.146 155,000 22,630` (không có "/KG"): SL 0,52 / 0,146 kg, chỉ nhận khi SL × đơn giá = thành tiền; trước đó ra SL 52 hoặc đọc ngược 155 × 146 đ | `src/extraction/line_items.py` |
| OCR lặp dấu, mất chữ số | Đọc được `210, ,000` và cân nặng `0, .460 KG`; khi đúng một dòng sai phép tính mà SL × đơn giá làm tổng các món khớp tổng in trên hóa đơn thì sửa dòng đó (`45 ,000 5, 000` → 45 000) | `src/extraction/line_items.py` |
| Rà 300 hóa đơn train (subagent) | Co.op "VAT05%", chữ CHA làm tên, số trong tên làm số lượng, "Tổng" lẫn trong dòng món (`docs/reports/review_train_300.md`) | `src/extraction/line_items.py` |
| Tổng tiền | Khi các món khớp một dòng "Tổng" in trên hóa đơn mà số model chọn không nằm trên dòng "Tổng" nào, dùng tổng các món. Tùy chọn `--total-from-items`, mặc định tắt | `src/extraction/total_check.py`, `extractor.py`, `extract_receipt.py`, `demo_app.py`, `decode_from_probs.py` |
| Khác | `.gitignore` chặn thư mục kết quả tải về (`phase1_eval/`, `kaggle/`, `final_eval/`, `eval/`); 23 unit test mới (317 test, đều qua) | `.gitignore`, `tests/` |

## Kết quả (validation)

| Chỉ số | Trước | Sau |
| --- | --- | --- |
| Tổng các món = tổng chuẩn | 62,8 % | **75,5 %** (sửa 12, hỏng 0) |
| Đọc được ít nhất một món | 85,1 % | 90,4 % |
| Tổng tiền đúng (bật `--total-from-items`) | 76,5 % | **81,6 %** (sửa 5, hỏng 0) |

Train, tổng các món = tổng chuẩn: 56,9 % → 74,1 %. Ảnh SunMart chụp thật (ngoài MC-OCR): đọc đúng cả 5 món.

## Cần biết

- Notebook 07 cell 7 sẽ báo LỆCH trên `main` (kiểm tra số test của luật cũ, 46 / 97); vì vậy notebook 07 được ghim vào commit
  `413599d`, nơi nó đã chạy. Báo cáo cuối vẫn đúng cho cấu hình đóng băng.
- Web demo (notebook 08) và web triển khai (`deploy/`) đã bật `total_from_items` và `quality_check` (cấu hình sản phẩm).
- Số lượng hàng cân là số thập phân (0.322).
- VietOCR tinh chỉnh "bịa chữ" ("Tổng thanh toán", "104 Trần Phú" lặp lại) ở khoảng 5 % ảnh không đọc được
  (train 53, validation 5, test 7): cần bước báo "ảnh không đọc được" thay vì đưa ra số bịa.

## Việc tiếp theo

1. ~~Nhận ra ảnh không đọc được~~ (xong: `quality_check`; ảnh bị xoay chưa xử lý).
2. Gõ tay tên hàng cho 30–50 hóa đơn để đo được độ đúng của tên hàng.
3. Phần lớn lỗi mặt hàng còn lại do OCR đọc sai số: cần OCR theo dòng (kế hoạch Tuần 5, mục 1.4).

## OCR

- **Đọc theo đoạn** (`--merge-lines`, notebook 09, Kaggle 19/11): chữ OCR sạch hơn (CER "trần OCR" validation: tổng tiền
  0,517 → 0,445, địa chỉ 0,198 → 0,155) nhưng **mặt hàng không đổi** (train 569 → 570, sửa 23 / hỏng 22; validation
  70 → 69) và OCR **chậm hơn 48 %** (2,39 → 3,53 s/ảnh). Không dùng; mặc định tắt.
- **Tinh chỉnh VietOCR cho chữ số** (notebook 10, Kaggle 19/11): 15 000 chữ số tổng hợp kiểu hóa đơn nhiệt
  (`src/recognition/synth_numbers.py`) + 991 con số của hóa đơn train `verified` (`src/recognition/number_pairs.py`) trộn
  với 23 151 cặp cũ. **Kết quả âm:** CER vùng CRAFT validation 5,68 % (vòng 1) → 6,20 % / 5,89 % ở epoch 1 / 2, dừng sớm,
  giữ VietOCR vòng 1; cổng 1 không qua nên không OCR lại, không train M1b v4. Mốc bộ đo chữ số (`scripts/eval_numbers.py`,
  VietOCR vòng 1, validation): **vùng TOTAL_COST có nhãn đúng chữ số 98,1 %** (514 vùng, CER 7,55 % chủ yếu ở phần chữ),
  dòng TOTAL_COST 92,4 % (224), số tổng hợp 96,1 % (1000). Nghĩa là VietOCR đã đọc chữ số gần đúng hết; lỗi tổng tiền và
  mặt hàng còn lại chủ yếu do CRAFT bỏ sót / cắt sai vùng, ảnh không đọc được và nhãn gốc, không do đọc nhầm số.
  Cùng kết luận với vòng 2 (18/11): tinh chỉnh thêm VietOCR trên dữ liệu này không có lợi.
- **Chặn bịa chữ** (`src/recognition/quality.py`, tùy chọn `--quality-check` của `extract_receipt.py` / `demo_app.py`,
  `ReceiptExtractor(quality_check=True)`, mặc định tắt): ảnh bị gắn cờ khi ≥ 30 % số từ thuộc cụm 3 từ lặp lại ≥ 3 lần
  (VietOCR tinh chỉnh viết lại "Tổng thanh toán", "104 Trần Phú", "10.000đ" trên ảnh không đọc được). Ngưỡng chọn trên
  train: gắn cờ 48 / 922 hóa đơn, cả 48 là ảnh bịa chữ, không hóa đơn thường nào; bỏ được 4 `verified` giả (món bịa cộng
  đúng "tổng" bịa, ví dụ 10 000 so với tổng thật 79 800). Validation (kiểm một lần): gắn cờ 5 / 115, cả 5 trước đây ra tổng
  tiền sai (10 000, 50 000, 55 104…), không mất tổng tiền đúng nào. Khi bị gắn cờ: đầu ra gọn chỉ còn
  `"warning": "unreadable"`, mặt hàng không bao giờ `verified`, không lấy tổng từ các món; web demo hiện "Ảnh khó đọc,
  hãy chụp lại".
