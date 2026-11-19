# Tiến độ sau báo cáo cuối (18–19/11/2026)

Nhánh `Thinh/items-line-grouping`, PR #20. Ưu tiên: **tổng tiền, tên hàng, đơn giá, số lượng** (người bán, địa chỉ, ngày giờ
không cải thiện thêm). Luật viết trên **train**, validation chấm một lần mỗi vòng, **test không chấm lại**: mọi số trong báo
cáo cuối giữ nguyên. Chi tiết và file nguồn: [`items_plan.md`](items_plan.md) mục 7–9.

## Đã làm

| Vòng | Sửa gì | File |
| --- | --- | --- |
| Mặt hàng bản 2 | Gom dòng theo độ nghiêng của ảnh; gộp số OCR tách đôi (`399` + `9,000`); giữ tên hàng dính vào dòng "Giá đăng ký" | `src/extraction/line_items.py` |
| Mặt hàng bản 3 | Hàng cân theo kg (`40 000/KG X 0,322 KG`: số lượng 0,322, không còn đọc cân nặng thành 322 đồng); nhận tiêu đề cột bị OCR đọc sai; bỏ qua dòng "Tổng" giả ở đầu hóa đơn; "giò heo" không bị bỏ | `src/extraction/line_items.py` |
| Số lượng thập phân | `0, 52 290,000 150,800` và `0.146 155,000 22,630` (không có "/KG"): SL 0,52 / 0,146 kg, chỉ nhận khi SL × đơn giá = thành tiền; trước đó ra SL 52 hoặc đọc ngược 155 × 146 đ | `src/extraction/line_items.py` |
| Tổng tiền | Khi các món khớp một dòng "Tổng" in trên hóa đơn mà số model chọn không nằm trên dòng "Tổng" nào, dùng tổng các món. Tùy chọn `--total-from-items`, mặc định tắt | `src/extraction/total_check.py`, `extractor.py`, `extract_receipt.py`, `demo_app.py`, `decode_from_probs.py` |
| Khác | `.gitignore` chặn thư mục kết quả tải về (`phase1_eval/`, `kaggle/`, `final_eval/`, `eval/`); 20 unit test mới (314 test, đều qua) | `.gitignore`, `tests/` |

## Kết quả (validation)

| Chỉ số | Trước | Sau |
| --- | --- | --- |
| Tổng các món = tổng chuẩn | 62,8 % | **73,4 %** (sửa 10, hỏng 0) |
| Đọc được ít nhất một món | 85,1 % | 90,4 % |
| Tổng tiền đúng (bật `--total-from-items`) | 76,5 % | **81,6 %** (sửa 5, hỏng 0) |

Train, tổng các món = tổng chuẩn: 56,9 % → 70,9 %.

## Cần biết

- Notebook 07 cell 7 sẽ báo LỆCH (kiểm tra số test của luật cũ, 46 / 97); báo cáo cuối vẫn đúng cho cấu hình đã nộp.
- Web demo (notebook 08) chưa bật `--total-from-items`; muốn bật thì thêm `total_from_items=True` ở cell 4.
- Số lượng hàng cân là số thập phân (0.322).
- VietOCR tinh chỉnh "bịa chữ" ("Tổng thanh toán", "104 Trần Phú" lặp lại) ở khoảng 5 % ảnh không đọc được
  (train 53, validation 5, test 7): cần bước báo "ảnh không đọc được" thay vì đưa ra số bịa.

## Việc tiếp theo

1. Nhận ra ảnh không đọc được hoặc bị xoay (cần Kaggle).
2. Gõ tay tên hàng cho 30–50 hóa đơn để đo được độ đúng của tên hàng.
3. Phần lớn lỗi mặt hàng còn lại do OCR đọc sai số: cần OCR theo dòng (kế hoạch Tuần 5, mục 1.4).
