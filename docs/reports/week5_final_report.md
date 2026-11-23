# Báo cáo Tuần 5 — Tích hợp cuối và tổng kết

*Nhóm QuetHoaDon, cập nhật 22/11/2026. Mọi con số dưới đây có file nguồn ghi ở mục 7. Có **hai cấu hình** và tập test MC-OCR
đã được dùng **hai lần**: ngày 17/11/2026 cho **cấu hình đóng băng** (cùng model, luật gốc) và ngày 22/11/2026 cho **cấu hình
sản phẩm** (bản chốt đang chạy trên web: cùng model, thêm luật mặt hàng bản 3, tổng tiền từ các món và chặn ảnh khó đọc).
Quy tắc của lần thứ hai được viết và commit trước khi chạy: [`final_test_protocol.md`](final_test_protocol.md). Mọi lựa chọn
trước hai lần chấm chỉ dựa trên train và validation.*

## 1. Tóm tắt

Hệ thống đọc ảnh hóa đơn tiếng Việt và trả về người bán, địa chỉ, ngày giờ, tổng tiền và từng mặt hàng. Pipeline:
tiền xử lý ảnh → CRAFT (tìm vùng chữ) → VietOCR **đã tinh chỉnh trên MC-OCR** (đọc chữ) → LayoutXLM **fine-tune** (gán nhãn
từng từ, model M1b v3) → luật ngày giờ và làm sạch SELLER / ADDRESS → đọc mặt hàng bằng luật.

Trên 114 hóa đơn test chấm được cả bốn trường, tỷ lệ đọc **đúng tuyệt đối cả bốn trường** tăng từ 9,6 % (model Tuần 4) lên
**18,4 %** (cấu hình đóng băng); cải thiện có ý nghĩa thống kê (McNemar p = 0,002, 10 hóa đơn chỉ bản mới đúng, 0 hóa đơn chỉ
bản cũ đúng). Nguồn cải thiện chính là OCR: số trường sai do chữ OCR giảm từ 95 xuống 52.

**Cấu hình sản phẩm (bản chốt, 22/11).** Cùng model và cùng OCR, thêm ba thay đổi ở khâu sau xử lý. Trên test: tổng tiền đúng
**71 → 80 / 98 (72,4 % → 81,6 %, McNemar p = 0,0039**, 9 hóa đơn chỉ bản mới đúng, 0 chỉ bản cũ đúng); cả bốn trường 18,4 % →
**21,1 %** (24 / 114, chưa đủ bằng chứng, p = 0,25); mặt hàng: tổng các món bằng tổng chuẩn **47,4 % → 74,2 %**; người bán, địa
chỉ, ngày, giờ không đổi. Việc chặn 5 ảnh khó đọc không làm mất kết quả đúng nào. Chi tiết ở mục 3.1 và 3.4.

## 2. Demo: Data → Model → Output

- **Data:** một ảnh hóa đơn gốc (JPG).
- **Model:** `ReceiptExtractor` (`src/pipeline/extractor.py`) chạy tiền xử lý, CRAFT, VietOCR tinh chỉnh, M1b v3, rồi các
  luật sau giải mã, bộ đọc mặt hàng, tổng tiền đối chiếu với các món và cờ "ảnh khó đọc" (cấu hình sản phẩm).
- **Output:** JSON gọn; trường nào không đọc được thì không xuất hiện; `items_check.status = "verified"` khi tổng các món
  bằng tổng in trên hóa đơn.

```bash
python scripts/extract_receipt.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
    --nested-threshold 0.5 --postprocess --items --total-from-items --quality-check --compact --image receipt.jpg
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

Ảnh khó đọc cho `{"file_id": "...", "fields": {}, "warning": "unreadable"}`: app yêu cầu chụp lại thay vì đưa số bịa.

**Web demo.** Có hai cách chạy, cùng một model (cấu hình sản phẩm: `total_from_items=True`, `quality_check=True`):

- **Web triển khai (`deploy/`, `docs/deploy.md`):** API trên một Hugging Face Space (CPU, trọng số lấy từ một repo riêng tư) và
  một trang web trên Vercel: người dùng chọn ảnh, trang thu nhỏ ảnh còn ≤ 2000 px, gửi tới `/api/extract` và hiện kết quả.
  Tốc độ trên Space (CPU miễn phí) **chưa được đo**; tài liệu triển khai chỉ có ước 10–30 giây mỗi ảnh.
- **Gradio trên Kaggle GPU (notebook 08, `scripts/demo_app.py`):** trang hiện ảnh sau tiền xử lý với khung từng từ tô màu theo
  trường model gán, bảng chỉ gồm các trường đọc được, bảng mặt hàng có cột đánh dấu dòng mà số lượng × đơn giá ≠ thành tiền,
  dòng kiểm tra tổng các món so với tổng in trên hóa đơn, JSON gọn và thời gian từng khâu; ảnh không đọc được cho thông báo
  "chụp lại". Ảnh JPEG tải lên được chuyển nguyên vẹn tới model. Cell 4 của notebook 08 kiểm tra đường đi của web cho ra
  đúng kết quả của đường đi đã chấm.

Con số hiệu năng của web là con số của cấu hình sản phẩm (mục 3.1); `verify_product_path.py` xác nhận cách chấm offline cho
đúng đầu ra của app trên 116 / 116 hóa đơn test.

Kết quả demo thật của **cấu hình đóng băng** nằm trong output cell 10 của `notebooks/07_final_evaluation_kaggle.ipynb` (notebook đã chạy đủ, mọi dòng KHỚP / PASS). Lưu ý trung thực: do danh sách chia tập có nhiều dòng cho mỗi ảnh, cell 10 chạy hai hóa đơn khác nhau (một hóa đơn lặp). Hóa đơn đầu chỉ đọc được tổng tiền; hóa đơn thứ hai đọc đủ bốn trường và 4 món, tổng các món 565 500 so với tổng 585 500 nên trạng thái `unverified`, và có một món đọc sai (số lượng 25, thành tiền 5 000). Đây là hạn chế thật, không chọn lại ví dụ đẹp hơn.
Đường đi của app được kiểm chứng bằng `scripts/verify_extractor.py`: chạy từng hóa đơn một, nó cho dự đoán giống hệt kết
quả đã chấm trên cả 115 hóa đơn validation (0 khác biệt, PASS).

## 3. Hiệu năng cuối

### 3.1 Bốn trường, tập test (đúng tuyệt đối sau chuẩn hóa), cấu hình đóng băng và cấu hình sản phẩm

| Trường | Tuần 4 + luật ngày giờ | Cấu hình đóng băng (17/11) | Chênh lệch ghép cặp [95 %] | McNemar p |
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

**Cấu hình sản phẩm (22/11), cùng 116 hóa đơn test, so với cấu hình đóng băng** (5 ảnh bị chặn khó đọc tính là không đọc được):

| Trường | Đóng băng (17/11) | Sản phẩm (22/11) | Chênh ghép cặp [95 %] | McNemar p |
| --- | --- | --- | --- | --- |
| SELLER | 77 / 107 (72,0 %) | 77 / 107 (72,0 %) | +0,0 | 1,000 |
| ADDRESS | 29 / 106 (27,4 %) | 29 / 106 (27,4 %) | +0,0 | 1,000 |
| Ngày và giờ cùng đúng | 77 / 106 | 77 / 106 | +0,0 | 1,000 |
| **Tổng tiền** | 71 / 98 (72,4 %) | **80 / 98 (81,6 %)** | **+9,2 [+4,1; +15,3]** | **0,0039** |
| **Cả bốn trường** | 21 / 114 (18,4 %) | **24 / 114 (21,1 %)** [14,6; 29,4] | +2,6 [+0,0; +6,1] | 0,25 |

Cải tiến nằm hoàn toàn ở tổng tiền: 11 hóa đơn lấy tổng từ các món, 9 chuyển từ sai sang đúng, không hóa đơn nào đổi theo chiều
xấu. Ở mọi trường, số hóa đơn "chỉ đóng băng đúng" là 0, nên việc chặn 5 ảnh khó đọc không làm mất kết quả đúng nào trên test.
Loại 5 ảnh đó ra, độ đúng là SELLER 77 / 103, ADDRESS 29 / 103, ngày 89 / 104, tổng tiền 80 / 95, cả bốn trường 24 / 110.
Tổng tiền trên validation là 80 / 98 (81,6 %) và trên test cũng 80 / 98: không có khoảng cách lạc quan. Đây là lần thứ hai tập
test được dùng; chi tiết quy tắc và số liệu ở [`final_test_protocol.md`](final_test_protocol.md).

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

Test, 97 hóa đơn có tổng chuẩn:

| | Luật đóng băng (17/11) | Luật bản 3 (sản phẩm, 22/11) |
| --- | --- | --- |
| Đọc được ít nhất một món | 80,4 % (78 / 97) | **92,8 %** (90 / 97) [85,8; 96,5] |
| Tổng các món bằng tổng chuẩn | 47,4 % (46 / 97) [37,8; 57,3] | **74,2 %** (72 / 97) [64,7; 81,9] |
| Báo `verified` và đúng | 42 / 44 (95,5 %) | 69 / 72 (95,8 %) [88,5; 98,6] |

Luật bản 3 (`src/extraction/line_items.py`) gom dòng theo độ nghiêng ảnh, ghép số OCR bị tách, đọc hàng cân theo kg (số lượng
thập phân), nhận tiêu đề cột bị đọc sai và sửa một dòng nhờ tổng in trên hóa đơn. Validation: 62,8 % → 74,5 % (test 74,2 %).
Khi `verified` thì đúng 95,8 %: người dùng có thể tin dòng `verified`. 3 hóa đơn bị chặn khó đọc không hiện món. Tên hàng không
chấm được.

### 3.5 Tốc độ (RQ3, Kaggle GPU T4, hóa đơn test)

| Khâu | Thời gian mỗi hóa đơn |
| --- | --- |
| Tiền xử lý | 0,11 s |
| CRAFT | 0,61 s |
| VietOCR | 1,72 s |
| **Tổng OCR** | **2,43 s** |
| LayoutXLM + giải mã, GPU | 58,9 ms (trung vị 54,8; p95 86,0; 60 hóa đơn) |
| LayoutXLM + giải mã, CPU | 913,2 ms (trung vị 908,2; p95 951,1; 10 hóa đơn) |

M1b v3 có 368,66 triệu tham số (1476 MB). Các số trên đo trên GPU (và CPU của máy Kaggle); tốc độ của web Hugging Face Space (CPU miễn phí) **chưa được đo**. Các thay đổi sau xử lý của cấu hình sản phẩm chỉ chạy trên chữ đã đọc nên không đổi các số này đáng kể (không đo riêng).

### 3.6 Kết quả âm (đã thử, không giữ)

| Thử nghiệm | Kết quả trên validation |
| --- | --- |
| 3 seed và ensemble trung bình xác suất | Không hơn một seed (F1 0,6636 / 0,6578 / 0,6613, ensemble 0,6627) |
| Tổng tiền = tiền khách đưa − tiền thối | Sửa 5, làm hỏng 5: không lợi |
| Tinh chỉnh VietOCR vòng 2 | CER 6,28 % → 6,56–6,98 %, dừng sớm: không qua cổng |
| Gán tên cửa hàng theo danh sách train | Không làm: 93 / 106 tên validation đã có trong train, sẽ chỉ là học thuộc |
| OCR đọc theo đoạn (gộp các hộp chữ chạm nhau, notebook 09) | Chữ OCR sạch hơn (CER trần OCR tổng tiền 0,517 → 0,445) nhưng mặt hàng không đổi và OCR chậm hơn 48 % (2,39 → 3,53 s mỗi ảnh): không dùng |
| Tinh chỉnh VietOCR cho chữ số (notebook 10) | CER vùng CRAFT validation 5,68 % → 6,20 % / 5,89 %, dừng sớm; VietOCR vòng 1 đã đọc đúng chữ số 98,1 % vùng TOTAL_COST nên lỗi còn lại chủ yếu do CRAFT bỏ sót hoặc cắt sai vùng và ảnh không đọc được |

## 4. Hạn chế và rủi ro

- **Độ chính xác còn thấp với địa chỉ:** ADDRESS đúng tuyệt đối 27,4 %; cả bốn trường đúng 21,1 % (cấu hình sản phẩm). App cần cho người dùng sửa.
- **Tập test dùng hai lần:** lần hai (22/11) cho cấu hình sản phẩm; luật mới được chọn trên train và validation, validation đã
  được chấm nhiều vòng. Hai cải tiến mới (tổng tiền, mặt hàng) cho cùng số trên validation và test, nhưng không còn tập
  chưa nhìn nào để đo thêm.
- **Nhãn gốc không nhất quán:** ví dụ "p.cẩm" và "p. cẩm", có nhãn gõ sai; một phần lỗi không thể sửa.
- **Chỉ đo trên MC-OCR:** chủ yếu chuỗi siêu thị, ảnh in rõ; chưa có tập ảnh chụp thật để đo sai lệch miền.
- **Mặt hàng:** tên hàng chưa có số đo; số món đọc đúng tổng mới 74,2 % trên test; luật chưa phủ giảm giá nhiều tầng và bảng
  không có tiêu đề; phần lớn lỗi còn lại do OCR đọc sai số.
- **Ảnh khó đọc:** khoảng 5 % ảnh (train 53 / 922, validation 5 / 115, test 5 / 116) VietOCR tinh chỉnh viết ra cụm từ bịa
  ("Tổng thanh toán", "104 Trần Phú") thay cho chữ thật. Cấu hình sản phẩm chặn các ảnh này và yêu cầu chụp lại, nhưng ngưỡng
  chặn chọn trên train nên có thể bỏ sót hoặc chặn nhầm ở dữ liệu khác.
- **Luật làm sạch viết sau khi xem lỗi validation:** số validation lạc quan; số test thì không.
- **Gần trùng giữa các tập:** khoảng 10 % hóa đơn validation / test có bản gần trùng trong train (Tuần 4: M1b test macro F1
  0,642 khi bỏ chúng).
- **Một lỗi phần mềm đã sửa:** tokenizer giữ chế độ đệm làm đường đi của app chia mỗi từ thành một cửa sổ; điểm chấm theo lô
  không bị ảnh hưởng; độ trễ đo trước khi sửa đã bỏ.
- **Tài nguyên:** model 1,4 GB; CPU mất khoảng 1 giây cho khâu LayoutXLM và vài giây cho OCR; web Space trên CPU miễn phí chưa đo tốc độ.

## 5. Hướng phát triển

1. Thêm dữ liệu OCR (ảnh hóa đơn thật, nhãn dòng) vì tinh chỉnh vòng 2 trên dữ liệu hiện có đã bão hòa.
2. Model mặt hàng học được: nhãn tự sinh từ các hóa đơn `verified` của train, kiểm tay một mẫu.
3. Điểm tin cậy cho từng trường để app chỉ hỏi lại khi không chắc (mới có cờ "ảnh khó đọc" cho cả hóa đơn).
4. Tập ảnh thật khoảng 100 hóa đơn, chấm một lần.
5. Phân loại nhóm chi tiêu (RQ2 ban đầu) khi có nhãn.
6. Thu nhỏ model (distillation hoặc lượng tử hóa) để chạy trên CPU.

## 6. Review GitHub

| Mục | Trạng thái |
| --- | --- |
| Mã nguồn | `src/` theo từng khâu; script trong `scripts/`; notebook 01–11 (08 là web demo, 11 là lần chấm test thứ hai) |
| Kiểm thử | 337 unit test, chạy không cần GPU (`python -m unittest discover -s tests -t .`), CI chạy trên mỗi PR |
| Tài liệu | `README.md`, `docs/architecture.md`, `docs/dataset.md`, `docs/training.md`, `docs/reports/` |
| Tái lập | Phiên bản cố định (Python 3.13, torch 2.11, transformers 5.20). Notebook 07 tính lại số của cấu hình đóng băng và in KHỚP / LỆCH, ghim vào commit `413599d` (luật mặt hàng trên `main` đã đổi sau đó, nên trên `main` cell 7 sẽ LỆCH); notebook 11 chạy trên `main` và tính lại cả hai cấu hình, có hai cổng tự kiểm |
| Trọng số | Không lưu trong repo (1,4 GB); nhóm giữ trong dataset Kaggle, Google Drive và một repo riêng tư trên Hugging Face (cho web); có lệnh huấn luyện lại |
| Notebook có output | Notebook 07 (cấu hình đóng băng) và notebook 11 (cấu hình sản phẩm) chạy bằng Save & Run All trên Kaggle; bản có output được thêm vào repo |

## 7. Nguồn số liệu

| Số liệu | File (thư mục `eval/` của các lần chạy Kaggle, bản sao trong `final_eval.zip`) |
| --- | --- |
| Bốn trường, test, cấu hình đóng băng | `decoded_v3_test.json`, `decoded_v2_test.json` (`final_eval.zip`) |
| Bốn trường, test, cấu hình sản phẩm | `decoded_product_test.json`, `paired_product_test.json`, `verify_product_test.json`, `final_test_summary.json` (`final_test.zip`, notebook 11) |
| So sánh ghép cặp | `paired_test.json` |
| Validation, cấu hình cuối | `decoded_v3_val.json` |
| Mặt hàng | `items_train.json`, `items_val.json`, `items_test.json` (luật đóng băng); `items_product_test.json` (luật bản 3) |
| Tinh chỉnh VietOCR | `vietocr_ft/finetune_report.json` |
| Tốc độ | `latency_final.json` (notebook 07, đo ngày 18/11) |
| RQ1 tiền xử lý | `ocr_ablation_ft.json`, `ocr_ablation_ft.md` |
| Kiểm chứng đường đi của app | `verify_extractor_val.json` |
| Hệ thống Tuần 4 | `week4_eval.zip` (`comparison_test.md`) |
