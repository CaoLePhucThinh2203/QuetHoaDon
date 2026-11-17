# Tài liệu học cho 3 thành viên — hiểu sản phẩm để trình bày báo cáo

*Soạn ngày 17/11/2026. Mỗi người học một phần; mọi con số bên dưới lấy từ file kết quả nêu kèm. Đừng trích số nào không có
file nguồn.*

## 0. Cả nhóm cần nắm trước (10 phút)

**Bài toán.** Đọc ảnh hóa đơn tiếng Việt và trích bốn thông tin: người bán (SELLER), địa chỉ (ADDRESS), thời gian
(TIMESTAMP), tổng tiền (TOTAL_COST). Mở rộng: từng mặt hàng (tên, số lượng, đơn giá, thành tiền). Dữ liệu: MC-OCR 2021,
chia theo hóa đơn: train 922, validation 115, test 116.

**Sơ đồ một dòng.** Ảnh → tiền xử lý → **CRAFT** (tìm vùng chữ) → **VietOCR** (đọc chữ) → **LayoutXLM** (gán nhãn từng từ theo
chữ và vị trí) → ghép từ thành 4 trường → luật ngày giờ và làm sạch SELLER / ADDRESS → đọc mặt hàng bằng luật.

**Ba kết luận chính của đề tài** (số test, chấm một lần, `eval/decoded_v3_test.json` và `decoded_v2_test.json`):
1. Fine-tune LayoutXLM (M1b) tốt hơn baseline B0 (0,658 so với 0,352 macro F1, Tuần 4).
2. Sai sót lớn nhất đến từ **chữ OCR đọc sai**, không phải từ bước gán nhãn. Tinh chỉnh VietOCR giảm lỗi OCR từ 95 xuống 52 trên
   test và nâng tỷ lệ đúng cả bốn trường từ 9,6 % lên 18,4 % (p = 0,002, ghép cặp).
3. Dữ liệu chỉ có nhãn bốn trường nên mặt hàng không huấn luyện được; làm bằng luật và chấm bằng tổng tiền chuẩn (47,4 % hóa
   đơn có tổng các món bằng tổng chuẩn).

**Tài liệu gốc để đọc:** `README.md`, `docs/architecture.md`, `docs/dataset.md`, `docs/training.md`,
`docs/reports/items_plan.md`.

---

## Phần A — Dữ liệu và tiền xử lý, OCR (người 1)

**Đọc:** `docs/dataset.md`, `src/preprocessing/`, `src/detection/`, `src/recognition/` (đặc biệt `finetune_data.py`,
`finetune.py`), `notebooks/01_mc_ocr_eda.ipynb`, `notebooks/06_phase1_ocr_kaggle.ipynb` (cell 6–9).

**Phải giải thích được:**
- Vì sao chia tập theo hóa đơn, và vì sao tập test chỉ được chấm một lần.
- Ba bước tiền xử lý (đổi cỡ, chỉnh nghiêng, tăng tương phản) và `configs/model.yaml`.
- CRAFT làm gì; vì sao có bộ lọc hộp chồng nhau (ngưỡng 0,8 và 0,5).
- Tinh chỉnh VietOCR: hai loại cặp huấn luyện (dòng chú thích với nhãn chính xác; vùng CRAFT gán nhãn bằng căn chỉnh
  Levenshtein), vì sao bỏ cặp có CER > 0,5, vì sao tự viết vòng huấn luyện thay cho `Trainer` của vietocr (imgaug hỏng với
  NumPy 2 và Trainer bỏ lô cuối).
- Vì sao vòng 2 thất bại (CER 6,28 % → 6,56–6,98 %, hàm mất mát đứng yên khoảng 0,50) và đó là kết quả âm hợp lệ.

**Số để trích:** CER vùng validation 13,03 % → 6,49 % (`vietocr_ft/finetune_report.json`); dòng đọc đúng hoàn toàn 66,8 % →
83,9 %; thời gian OCR 2,43 giây mỗi hóa đơn (tiền xử lý 0,11, CRAFT 0,61, VietOCR 1,72; `latency_final.json`).

**Câu hỏi hội đồng có thể hỏi:** Tại sao không chọn epoch theo tập test? (vì tập test chỉ để chấm cuối). Vì sao CER
trung bình SELLER của "trần OCR" tăng (0,143 → 0,192) dù số khớp tuyệt đối tăng (63 → 81 trên 106)? (`eval/ocr_runs_val.json`; chưa có giải thích đã kiểm chứng, nên nói rõ là chưa rõ).

---

## Phần B — Mô hình và huấn luyện, baseline (người 2)

**Đọc:** `docs/training.md`, `docs/architecture.md`, `src/baseline/b0.py`, `src/training/` (`windows.py`, `data.py`,
`engine.py`), `scripts/train_layoutxlm.py`, `notebooks/03_train_evaluate_kaggle.ipynb`.

**Phải giải thích được:**
- BIO tagging: 9 nhãn (O và B/I của 4 trường); vì sao chia hóa đơn dài thành cửa sổ 480 sub-token, chồng 100.
- Ba hệ thống: B0 (đặc trưng đóng băng + K-means + luật), M1a (LayoutXLM đóng băng + lớp tuyến tính), M1b (fine-tune toàn bộ).
- Vì sao dùng trọng số lớp `sqrt_inverse` (nhãn O áp đảo) và chọn checkpoint theo macro F1.
- **Lỗi đã tìm ra và sửa trong tuần 5:** sau khi processor chạy với `padding="max_length"`, tokenizer nhanh giữ chế độ đệm
  512 nên mọi từ bị đếm 512 sub-token và một hóa đơn bị chia thành cả trăm cửa sổ khi chạy từng hóa đơn (đường đi của app).
  Điểm chấm theo lô không bị ảnh hưởng; độ trễ đo lần đầu (4586 ms) là số của lỗi, sau sửa là 58,9 ms. `verify_extractor.py`
  xác nhận đường đi của app tái lập đúng dự đoán Tuần 4 (115/115 hóa đơn).
- Kết quả âm: 3 seed và ensemble không cải thiện (lenient macro F1 0,6636 / 0,6578 / 0,6613 / 0,6627).

**Số để trích:** macro F1 test B0 0,352, M1a 0,638, M1b 0,658; M1b có 368,66 triệu tham số, 1476 MB; khâu LayoutXLM 58,9 ms
GPU (p95 86,0) và 913,2 ms CPU (`eval/latency_final.json`, notebook 07 cell 9).

**Câu hỏi có thể hỏi:** Vì sao không dùng CRF? Vì sao đóng băng bộ mã hóa (M1a) lại kém hơn? M1b v3 khác v2 ở đâu? (huấn
luyện trên chữ OCR của VietOCR tinh chỉnh, cùng cấu hình).

---

## Phần C — Đánh giá, lỗi, mặt hàng và câu hỏi nghiên cứu (người 3)

**Đọc:** `src/evaluation/app_metrics.py`, `scripts/decode_from_probs.py`, `scripts/compare_strict.py`,
`src/extraction/datetime_rules.py`, `field_cleanup.py`, `line_items.py`, `scripts/score_items.py`,
`docs/reports/items_plan.md`.

**Phải giải thích được:**
- Hai cách chấm: **lenient** (chấp nhận sai tới 30 % ký tự, dùng ở Tuần 4) và **strict** (chuỗi phải bằng nhau sau chuẩn hóa,
  là cái người dùng app thấy). Khoảng tin cậy Wilson; kiểm định McNemar chính xác và bootstrap theo hóa đơn.
- Vì sao luật ngày giờ và làm sạch cải thiện (giữ giờ cạnh ngày, bỏ từ lặp do hộp chồng nhau, bỏ nhãn "Tel:" và số điện thoại
  ở cuối) và vì sao **không** gán tên cửa hàng theo danh sách train (93/106 tên validation đã có trong train: đó là học thuộc).
- Rủi ro quá khớp: luật làm sạch viết sau khi xem lỗi validation, nên số validation lạc quan; số test thì không.
- Đọc mặt hàng: tách dòng, khoanh bảng hàng, số lượng × đơn giá = thành tiền, tổng các món so với tổng chuẩn; vì sao tên hàng
  không chấm được; nhãn `verified` đúng 95,5 % (42/44) trên test.

**Số để trích (test, `eval/decoded_*_test.json`, `eval/paired_test.json`, `eval/items_test.json`):**

| Đúng tuyệt đối | Tuần 4 | Cuối cùng | Chênh (ghép cặp) |
| --- | --- | --- | --- |
| SELLER | 61/107 | 77/107 | +15,0 điểm, p = 0,0004 |
| ADDRESS | 17/106 | 29/106 | +11,3, p = 0,0005 |
| Ngày và giờ | 59/106 | 77/106 | +17,0, p < 0,0001 |
| Tổng tiền | 64/98 | 71/98 | +7,1, p = 0,065 (chưa có ý nghĩa) |
| Cả bốn trường | 11/114 | 21/114 | +8,8, p = 0,002 |

Mặt hàng (test, 97 hóa đơn có tổng chuẩn): đọc được ít nhất một món 80,4 %; tổng các món = tổng chuẩn 47,4 % (46/97).

**Giới hạn phải nói thẳng khi trình bày:** ADDRESS đúng mới 27,4 %; cả bốn trường đúng 18,4 %; nhãn gốc có lỗi và mâu thuẫn;
tên hàng chưa có số đo; chưa thử trên ảnh chụp thật ngoài MC-OCR.

**Câu hỏi có thể hỏi:** Vì sao tổng tiền cải thiện nhưng p chỉ 0,065? (chỉ 11 hóa đơn thay đổi: 9 tốt hơn, 2 xấu hơn). Có dùng
tập test để chọn cấu hình không? (không; mọi lựa chọn trên validation, test chấm một lần với cấu hình đóng băng).

---

## Phần chung: hướng phát triển (đưa vào báo cáo)

1. Tinh chỉnh OCR bằng nhiều dữ liệu hơn hoặc dữ liệu hóa đơn thật; vòng 2 hiện bão hòa.
2. Huấn luyện mô hình riêng cho mặt hàng bằng nhãn tự sinh từ các hóa đơn có `verified` (cần kiểm tay một mẫu).
3. Bộ ảnh chụp thật của nhóm để đo sai lệch miền.
4. Điểm tin cậy cho từng trường để app nhắc người dùng kiểm tra lại.
5. Phân loại nhóm chi tiêu (RQ2 của tài liệu gốc) chưa làm vì chưa có nhãn.
