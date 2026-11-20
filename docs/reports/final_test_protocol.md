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

*Điền sau khi chạy (ngày, commit, link notebook có output, bảng số). Không sửa các mục 1–6.*
