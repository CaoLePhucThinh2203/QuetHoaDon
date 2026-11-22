# Giao thức chấm test lần 2: cấu hình sản phẩm (viết trước khi chạy)

*Viết ngày 21/11/2026, **trước** khi chạy. Commit chứa file này có trước lần chạy trên Kaggle; lịch sử Git là bằng chứng
mọi quy tắc dưới đây được đặt trước khi nhìn kết quả. Chạy bằng `notebooks/11_final_test_product_kaggle.ipynb`.*

## 1. Vì sao chấm test thêm một lần

Tập test MC-OCR đã được chấm một lần ngày 17/11/2026 cho **cấu hình đóng băng** (báo cáo cuối
`week5_final_report.md`). Sau đó nhóm chọn **cấu hình sản phẩm** (bản web đang chạy, `deploy/api/app.py`) làm bản chốt.
Bản này chỉ có số validation, mà validation đã được dùng nhiều vòng để chỉnh luật nên lạc quan. Lần chấm này cho bản chốt
một con số test. Đây là **lần thứ hai** nhìn tập test; báo cáo ghi rõ điều đó.

## 2. Cấu hình được chấm (không đổi gì thêm)

| Thành phần | Giá trị |
| --- | --- |
| Model | M1b v3, run `m1b_v3_ft_n05_20261117_010041` (dataset Kaggle `phase1-result`) — **giống cấu hình đóng băng** |
| OCR | Tiền xử lý → CRAFT, lọc hộp chồng 0,5 → VietOCR tinh chỉnh vòng 1 (`vietocr_finetuned.pth`) — **giống** |
| Sau giải mã | Luật ngày giờ + làm sạch SELLER / ADDRESS — **giống** |
| Đọc mặt hàng | Luật bản 3 của 19/11 (`src/extraction/line_items.py` trên `main`, commit `3a14cc4`) — **mới** |
| Tổng tiền | `total_from_items=True` (`src/extraction/total_check.py`) — **mới** |
| Ảnh khó đọc | `quality_check=True`, ngưỡng 0,3 chọn trên train (`src/recognition/quality.py`) — **mới** |

Mã quyết định (luật, ngưỡng) là của commit `3a14cc4`. Nhánh chứa giao thức này chỉ **thêm** công cụ chấm
(`--quality-check` cho `decode_from_probs.py` và `score_items.py`, `scripts/verify_product_path.py`), không sửa luật nào.

Dữ liệu: 116 hóa đơn test, token dataset `tok_ft_n05` (chữ OCR của chính cấu hình OCR trên), như lần chấm 17/11.

## 3. Cách tính

- **Ảnh bị gắn cờ khó đọc** thì app không đưa trường nào (chỉ báo "chụp lại"). Khi chấm: **không có dự đoán**, mọi trường
  có nhãn của hóa đơn đó tính là **sai**; mặt hàng tính là không đọc được. Đây là cách tính bất lợi cho mình, chọn trước.
- **Chỉ số chính:** đúng tuyệt đối (strict) từng trường SELLER, ADDRESS, ngày, giờ, tổng tiền và cả bốn trường; so sánh
  ghép cặp với cấu hình đóng băng trên cùng hóa đơn (McNemar chính xác, khoảng bootstrap 95 %, `compare_strict.py`).
- **Mặt hàng:** tổng các món = tổng chuẩn; trong số `verified`, bao nhiêu đúng (`score_items.py --quality-check`).
- **Phụ (không phải chỉ số chính):** số ảnh bị gắn cờ, số tổng tiền lấy từ các món, độ đúng trên các ảnh không bị gắn cờ.

## 4. Số đã biết trước khi chạy (validation, đã dùng để chỉnh luật nên lạc quan)

| | Cấu hình đóng băng | Cấu hình sản phẩm |
| --- | --- | --- |
| SELLER | 74 / 106 | 74 / 106 |
| ADDRESS | 29 / 112 | 29 / 112 |
| Ngày | 79 / 99 | 78 / 99 (1 ảnh bị gắn cờ có ngày đúng) |
| Giờ hh:mm | 54 / 80 | 54 / 80 |
| Tổng tiền | 75 / 98 | 80 / 98 |
| Cả bốn trường | 20 / 115 | 21 / 115 |
| Tổng các món = tổng chuẩn | 59 / 94 | 70 / 94 |
| Ảnh bị gắn cờ | — | 5 / 115 |

Kiểm chứng trên validation (21/11, Mac, không cần GPU): chấm offline cho ra **đúng** đầu ra của app
(`verify_product_path.py`: 115 / 115 giống hệt, PASS).

## 5. Quy tắc chạy

1. **Hai cổng phải qua trước khi xem số của cấu hình sản phẩm** (cell 5 và 6 của notebook 11):
   - tính lại cấu hình đóng băng trên test phải ra đúng số đã báo cáo 17/11 (77 / 29 / 89 / 61 / 71 / 21): xác nhận xác suất
     và dữ liệu không đổi;
   - `verify_product_path.py` trên test phải PASS: số chấm offline là số của app.
   Nếu một cổng không qua thì **dừng, không chạy cell 7**, tìm lỗi rồi chạy lại từ đầu.
2. Cell 7 (chấm cấu hình sản phẩm) **chạy đúng một lần**. Nếu nó lỗi vì phần mềm (không ra số nào), được sửa lỗi chạy và
   chạy lại; không được đổi luật, ngưỡng hay cách tính.
3. **Công bố kết quả dù tốt hay xấu**, kể cả khi thấp hơn cấu hình đóng băng.
4. Sau lần chạy này **không chỉnh luật để tăng số trên MC-OCR nữa**: muốn cải tiến thêm phải có dữ liệu chấm mới.

## 6. Cách ghi vào báo cáo

- Giữ nguyên số của cấu hình đóng băng (17/11) trong `week5_final_report.md`.
- Thêm mục "Cấu hình sản phẩm" với số test của lần chạy này, ngày chạy, commit, bảng ghép cặp với cấu hình đóng băng, và
  câu: "đây là lần thứ hai tập test được dùng; luật mới được chọn trên train và validation".

## 7. Kết quả

*Chạy ngày 22/11/2026 trên Kaggle (GPU T4), notebook 11, mã nguồn `main` commit `d3dd2d8`. Một lần chạy; mục 1–6 không bị sửa.*

**Hai cổng qua.** Cổng 1: tính lại cấu hình đóng băng trên test ra đúng 77 / 29 / 89 / 61 / 71 / 21 (KHỚP). Cổng 2:
`verify_product_path.py` trên test, 116 / 116 hóa đơn giống hệt đầu ra của app, 0 khác biệt, PASS.

**Đúng tuyệt đối trên test (cấu hình sản phẩm; 5 ảnh bị chặn tính là không đọc được):**

| Trường | Đóng băng (17/11) | Sản phẩm | Chênh ghép cặp [95 %] | Chỉ đóng băng đúng | Chỉ sản phẩm đúng | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| SELLER | 77 / 107 (72,0 %) | 77 / 107 (72,0 %) | +0,0 điểm | 0 | 0 | 1,000 |
| ADDRESS | 29 / 106 (27,4 %) | 29 / 106 (27,4 %) | +0,0 | 0 | 0 | 1,000 |
| Ngày | 89 / 106 (84,0 %) | 89 / 106 (84,0 %) | | | | |
| Giờ hh:mm | 61 / 85 (71,8 %) | 61 / 85 (71,8 %) | | | | |
| Ngày và giờ cùng đúng | 77 / 106 | 77 / 106 | +0,0 | 0 | 0 | 1,000 |
| **Tổng tiền** | 71 / 98 (72,4 %) | **80 / 98 (81,6 %)** [72,8; 88,1] | **+9,2 [+4,1; +15,3]** | 0 | 9 | **0,0039** |
| **Cả bốn trường** | 21 / 114 (18,4 %) | **24 / 114 (21,1 %)** [14,6; 29,4] | +2,6 [+0,0; +6,1] | 0 | 3 | 0,25 |

- **Tổng tiền:** 11 hóa đơn được lấy tổng từ các món; 9 trong số đó chuyển từ sai sang đúng, không hóa đơn nào chuyển từ đúng
  sang sai. Cải thiện có ý nghĩa thống kê.
- **Cả bốn trường:** +3 hóa đơn, chưa đủ bằng chứng (p = 0,25).
- **Ảnh bị chặn khó đọc:** 5 / 116 (`145013kcxya`, `145014ebrwa`, `145014ecchz`, `145014exzpp`, `145014motth`). Ở **mọi** trường,
  số hóa đơn "chỉ đóng băng đúng" là 0: việc chặn không làm mất một kết quả đúng nào trên test. Loại 5 ảnh này ra, độ đúng
  là SELLER 77 / 103, ADDRESS 29 / 103, ngày 89 / 104, tổng tiền 80 / 95, cả bốn trường 24 / 110.
- Nguyên nhân sai (đóng băng → sản phẩm): chữ OCR 52 → 52, mô hình gán sai 31 → 25, mô hình bỏ sót 5 → 2.

**Mặt hàng (test, 97 hóa đơn có tổng chuẩn; luật bản 3, 3 hóa đơn bị chặn không hiện món):**

| | Luật đóng băng (17/11) | Luật bản 3 (sản phẩm) |
| --- | --- | --- |
| Đọc được ít nhất một món | 78 / 97 (80,4 %) | **90 / 97 (92,8 %)** [85,8; 96,5] |
| Tổng các món = tổng chuẩn | 46 / 97 (47,4 %) | **72 / 97 (74,2 %)** [64,7; 81,9] |
| Từ hai món trở lên | 58 / 97 | 61 / 97 |
| `verified` (tổng các món = tổng in) | 44 / 97 | 72 / 97 (74,2 %) |
| Trong số `verified`, bằng tổng chuẩn | 42 / 44 (95,5 %) | **69 / 72 (95,8 %)** [88,5; 98,6] |
| Dòng có SL × đơn giá = thành tiền | 204 / 222 (91,9 %) | 255 / 279 (91,4 %) |

**So với validation:** tổng tiền validation 80 / 98 (81,6 %) và test 80 / 98 (81,6 %); mặt hàng validation 74,5 % và test
74,2 %. Không có khoảng cách lạc quan giữa validation và test cho hai cải tiến này.

**Hạn chế phải ghi cùng:** đây là lần thứ hai tập test được dùng; luật mới được chọn trên train và validation (validation
đã được chấm nhiều vòng). Tên hàng vẫn không chấm được. SELLER, ADDRESS, ngày và giờ **không đổi** so với cấu hình đóng băng:
cải tiến nằm ở tổng tiền và mặt hàng.

Nguồn: `final_test.zip` (`eval/decoded_product_test.json`, `paired_product_test.json`, `items_product_test.json`,
`verify_product_test.json`, `final_test_summary.json`) và notebook 11 có output.
