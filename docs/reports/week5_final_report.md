# Báo cáo Tuần 5 — Tích hợp cuối và tổng kết

*Nhóm QuetHoaDon, 18/11/2026. Mọi con số dưới đây có file nguồn ghi ở mục 7. Tập test MC-OCR được chấm **một lần** ngày
17/11/2026 với cấu hình đã đóng băng; mọi lựa chọn trước đó chỉ dựa trên validation.*

## 1. Tóm tắt

Hệ thống đọc ảnh hóa đơn tiếng Việt và trả về người bán, địa chỉ, ngày giờ, tổng tiền và từng mặt hàng. Pipeline:
tiền xử lý ảnh → CRAFT (tìm vùng chữ) → VietOCR **đã tinh chỉnh trên MC-OCR** (đọc chữ) → LayoutXLM **fine-tune** (gán nhãn
từng từ, model M1b v3) → luật ngày giờ và làm sạch SELLER / ADDRESS → đọc mặt hàng bằng luật.

Trên 114 hóa đơn test chấm được cả bốn trường, tỷ lệ đọc **đúng tuyệt đối cả bốn trường** tăng từ 9,6 % (model Tuần 4) lên
**18,4 %**; cải thiện có ý nghĩa thống kê (McNemar p = 0,002, 10 hóa đơn chỉ bản mới đúng, 0 hóa đơn chỉ bản cũ đúng). Nguồn
cải thiện chính là OCR: số trường sai do chữ OCR giảm từ 95 xuống 52.

## 2. Demo: Data → Model → Output

- **Data:** một ảnh hóa đơn gốc (JPG).
- **Model:** `ReceiptExtractor` (`src/pipeline/extractor.py`) chạy tiền xử lý, CRAFT, VietOCR tinh chỉnh, M1b v3, rồi các
  luật sau giải mã và bộ đọc mặt hàng.
- **Output:** JSON gọn; trường nào không đọc được thì không xuất hiện; `items_check.status = "verified"` khi tổng các món
  bằng tổng in trên hóa đơn.

```bash
python scripts/extract_receipt.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
    --nested-threshold 0.5 --postprocess --items --compact --image receipt.jpg
```

Dạng đầu ra (khung, không phải kết quả thật):

```json
{"file_id": "...",
 "fields": {"TIMESTAMP": {"text": "...", "date": "dd/mm/yyyy", "time": "hh:mm"},
            "TOTAL_COST": {"text": "...", "amount_vnd": 0},
            "SELLER": {"text": "..."}, "ADDRESS": {"text": "..."}},
 "items": [{"name": "...", "quantity": 1, "unit_price": 0, "amount": 0}],
 "items_check": {"sum": 0, "status": "verified", "predicted_total": 0}}
```

Kết quả demo thật nằm trong output cell 10 của `notebooks/07_final_evaluation_kaggle.ipynb` (notebook đã chạy đủ, mọi dòng KHỚP / PASS). Lưu ý trung thực: do danh sách chia tập có nhiều dòng cho mỗi ảnh, cell 10 chạy hai hóa đơn khác nhau (một hóa đơn lặp). Hóa đơn đầu chỉ đọc được tổng tiền; hóa đơn thứ hai đọc đủ bốn trường và 4 món, tổng các món 565 500 so với tổng 585 500 nên trạng thái `unverified`, và có một món đọc sai (số lượng 25, thành tiền 5 000). Đây là hạn chế thật, không chọn lại ví dụ đẹp hơn.
Đường đi của app được kiểm chứng bằng `scripts/verify_extractor.py`: chạy từng hóa đơn một, nó cho dự đoán giống hệt kết
quả đã chấm trên cả 115 hóa đơn validation (0 khác biệt, PASS).

## 3. Hiệu năng cuối

### 3.1 Bốn trường, tập test (đúng tuyệt đối sau chuẩn hóa)

| Trường | Tuần 4 + luật ngày giờ | Cấu hình cuối | Chênh lệch ghép cặp [95 %] | McNemar p |
| --- | --- | --- | --- | --- |
| SELLER | 61 / 107 (57,0 %) | 77 / 107 (72,0 %) | +15,0 điểm [+7,5; +22,4] | 0,0004 |
| ADDRESS | 17 / 106 (16,0 %) | 29 / 106 (27,4 %) | +11,3 [+5,7; +17,9] | 0,0005 |
| Ngày | 83 / 106 (78,3 %) | 89 / 106 (84,0 %) | — | — |
| Giờ hh:mm | 43 / 85 (50,6 %) | 61 / 85 (71,8 %) | — | — |
| Ngày và giờ cùng đúng | 59 / 106 | 77 / 106 | +17,0 [+9,4; +24,5] | < 0,0001 |
| Tổng tiền | 64 / 98 (65,3 %) | 71 / 98 (72,4 %) | +7,1 [+1,0; +14,3] | 0,065 |
| **Cả bốn trường** | 11 / 114 (9,6 %) | **21 / 114 (18,4 %)** | +8,8 [+3,5; +14,0] | 0,002 |
| F1 chịu lỗi OCR (macro) | 0,664 | 0,780 | | |

Tổng tiền tăng nhưng chưa đủ bằng chứng thống kê (chỉ 11 hóa đơn đổi kết quả: 9 tốt hơn, 2 xấu hơn).

### 3.2 So sánh các hệ thống (Tuần 4, F1 chịu lỗi OCR, test)

| Hệ thống | Macro F1 [95 %] | Đúng cả bốn trường |
| --- | --- | --- |
| B0: đặc trưng LayoutXLM đóng băng + phân cụm + luật | 0,352 [0,32; 0,39] | 0 / 116 |
| M1a: LayoutXLM đóng băng + lớp tuyến tính | 0,638 [0,58; 0,69] | 30 / 116 |
| M1b: LayoutXLM fine-tune | 0,658 [0,60; 0,71] | 32 / 116 |

### 3.3 OCR (RQ1)

- Tinh chỉnh VietOCR, validation, vùng CRAFT: CER 13,03 % → 6,49 % (chênh ghép cặp −6,54 điểm [−7,60; −5,65]); dòng đọc đúng
  hoàn toàn 66,8 % → 83,9 %.
- Thí nghiệm tiền xử lý với VietOCR tinh chỉnh (validation, 668 vùng có nhãn, CER %; chênh ghép cặp so với "chỉ đổi cỡ", khoảng 95 % bootstrap):

| Ảnh | Chỉ đổi cỡ | + chỉnh nghiêng | + tăng tương phản | Cả hai |
| --- | --- | --- | --- | --- |
| Gốc | 9,7 | 10,1 (+0,37 [−0,77; +1,73]) | 9,7 (+0,02 [−0,17; +0,24]) | 9,8 (+0,08 [−0,82; +0,97]) |
| Làm mờ | 43,3 | 42,0 | 44,0 | 43,7 |
| Tối | 10,4 | 10,3 | 10,0 | 10,5 |
| Nhiễu | 11,3 | 11,2 | 12,4 (+1,11 [+0,50; +1,74]) | 12,2 |

  Kết luận: với VietOCR đã tinh chỉnh, chỉnh nghiêng và tăng tương phản **không** giảm CER có ý nghĩa; chỉ một trường hợp
  khác biệt rõ và theo hướng xấu đi (tăng tương phản trên ảnh nhiễu). Làm mờ là nhiễu loạn gây hại nhất (CER 43 %), tiền
  xử lý không cứu được. Giữ tiền xử lý trong pipeline vì cấu hình đóng băng đã dùng nó cho mọi số liệu; thí nghiệm này không
  dùng để chọn lại.

### 3.4 Mặt hàng (không có nhãn mặt hàng; chấm bằng tổng tiền chuẩn)

Test, 97 hóa đơn có tổng chuẩn: đọc được ít nhất một món 80,4 %; tổng các món bằng tổng chuẩn **47,4 %** (46 / 97,
[37,8; 57,3]); khi hệ thống báo `verified` thì đúng 42 / 44 (95,5 %). Validation: 62,8 % và 98,1 %. Tên hàng không chấm được.

### 3.5 Tốc độ (RQ3, Kaggle GPU T4, hóa đơn test)

| Khâu | Thời gian mỗi hóa đơn |
| --- | --- |
| Tiền xử lý | 0,11 s |
| CRAFT | 0,61 s |
| VietOCR | 1,72 s |
| **Tổng OCR** | **2,43 s** |
| LayoutXLM + giải mã, GPU | 58,9 ms (trung vị 54,8; p95 86,0; 60 hóa đơn) |
| LayoutXLM + giải mã, CPU | 913,2 ms (trung vị 908,2; p95 951,1; 10 hóa đơn) |

M1b v3 có 368,66 triệu tham số (1476 MB).

### 3.6 Kết quả âm (đã thử, không giữ)

| Thử nghiệm | Kết quả trên validation |
| --- | --- |
| 3 seed và ensemble trung bình xác suất | Không hơn một seed (F1 0,6636 / 0,6578 / 0,6613, ensemble 0,6627) |
| Tổng tiền = tiền khách đưa − tiền thối | Sửa 5, làm hỏng 5: không lợi |
| Tinh chỉnh VietOCR vòng 2 | CER 6,28 % → 6,56–6,98 %, dừng sớm: không qua cổng |
| Gán tên cửa hàng theo danh sách train | Không làm: 93 / 106 tên validation đã có trong train, sẽ chỉ là học thuộc |

## 4. Hạn chế và rủi ro

- **Độ chính xác còn thấp với địa chỉ:** ADDRESS đúng tuyệt đối 27,4 %; cả bốn trường đúng 18,4 %. App cần cho người dùng sửa.
- **Nhãn gốc không nhất quán:** ví dụ "p.cẩm" và "p. cẩm", có nhãn gõ sai; một phần lỗi không thể sửa.
- **Chỉ đo trên MC-OCR:** chủ yếu chuỗi siêu thị, ảnh in rõ; chưa có tập ảnh chụp thật để đo sai lệch miền.
- **Mặt hàng:** tên hàng chưa có số đo; luật chưa phủ giảm giá nhiều tầng, hàng cân theo kg, bảng không có tiêu đề.
- **Luật làm sạch viết sau khi xem lỗi validation:** số validation lạc quan; số test thì không.
- **Gần trùng giữa các tập:** khoảng 10 % hóa đơn validation / test có bản gần trùng trong train (Tuần 4: M1b test macro F1
  0,642 khi bỏ chúng).
- **Một lỗi phần mềm đã sửa:** tokenizer giữ chế độ đệm làm đường đi của app chia mỗi từ thành một cửa sổ; điểm chấm theo lô
  không bị ảnh hưởng; độ trễ đo trước khi sửa đã bỏ.
- **Tài nguyên:** model 1,4 GB; CPU mất khoảng 1 giây cho khâu LayoutXLM và vài giây cho OCR.

## 5. Hướng phát triển

1. Thêm dữ liệu OCR (ảnh hóa đơn thật, nhãn dòng) vì tinh chỉnh vòng 2 trên dữ liệu hiện có đã bão hòa.
2. Model mặt hàng học được: nhãn tự sinh từ các hóa đơn `verified` của train, kiểm tay một mẫu.
3. Điểm tin cậy cho từng trường để app chỉ hỏi lại khi không chắc.
4. Tập ảnh thật khoảng 100 hóa đơn, chấm một lần.
5. Phân loại nhóm chi tiêu (RQ2 ban đầu) khi có nhãn.
6. Thu nhỏ model (distillation hoặc lượng tử hóa) để chạy trên CPU.

## 6. Review GitHub

| Mục | Trạng thái |
| --- | --- |
| Mã nguồn | `src/` theo từng khâu; script trong `scripts/`; notebook 01–07 |
| Kiểm thử | 279 unit test, chạy không cần GPU (`python -m unittest discover -s tests -t .`), CI chạy trên mỗi PR |
| Tài liệu | `README.md`, `docs/architecture.md`, `docs/dataset.md`, `docs/training.md`, `docs/reports/` |
| Tái lập | Phiên bản cố định (Python 3.13, torch 2.11, transformers 5.20); notebook 07 tính lại mọi số và in KHỚP / LỆCH |
| Trọng số | Không lưu trong repo (1,4 GB); nhóm giữ trong dataset Kaggle và Google Drive; có lệnh huấn luyện lại |
| Notebook có output | Notebook 07 chạy bằng Save & Run All trên Kaggle; bản có output được thêm vào repo |

## 7. Nguồn số liệu

| Số liệu | File (thư mục `eval/` của các lần chạy Kaggle, bản sao trong `final_eval.zip`) |
| --- | --- |
| Bốn trường, test | `decoded_v3_test.json`, `decoded_v2_test.json` |
| So sánh ghép cặp | `paired_test.json` |
| Validation, cấu hình cuối | `decoded_v3_val.json` |
| Mặt hàng | `items_train.json`, `items_val.json`, `items_test.json` |
| Tinh chỉnh VietOCR | `vietocr_ft/finetune_report.json` |
| Tốc độ | `latency_final.json` (notebook 07, đo ngày 18/11) |
| RQ1 tiền xử lý | `ocr_ablation_ft.json`, `ocr_ablation_ft.md` |
| Kiểm chứng đường đi của app | `verify_extractor_val.json` |
| Hệ thống Tuần 4 | `week4_eval.zip` (`comparison_test.md`) |
