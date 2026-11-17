# Kế hoạch mở rộng — đọc từng mặt hàng (tên, số lượng, giá)

*Soạn ngày 17/11/2026, sau khi chấm tập test một lần với cấu hình đã đóng băng. Hệ thống 4 trường (SELLER, ADDRESS,
TIMESTAMP, TOTAL_COST) **giữ nguyên**: các con số test bên dưới là kết quả chính của đề tài và không được làm thay đổi.
Phần mặt hàng là **một module mới chạy sau**, có tập đánh giá và số liệu riêng.*

## 1. Mốc hiện tại (đã kiểm chứng trên Kaggle)

Tập test MC-OCR, 116 hóa đơn, chấm **một lần** (`eval/decoded_v3_test.json`, `eval/decoded_v2_test.json`).
Cấu hình: VietOCR tinh chỉnh + lọc hộp chồng 0.5 → M1b v3 (run `m1b_v3_ft_n05_20261117_010041`) → luật ngày giờ → hậu xử lý
SELLER/ADDRESS.

| Đọc đúng tuyệt đối (test) | M1b Tuần 4 + luật ngày giờ | Cấu hình cuối |
| --- | --- | --- |
| SELLER | 61 / 107 (57,0 %) | 77 / 107 (72,0 %) |
| ADDRESS | 17 / 106 (16,0 %) | 29 / 106 (27,4 %) |
| Ngày | 83 / 106 (78,3 %) | 89 / 106 (84,0 %) |
| Giờ hh:mm | 43 / 85 (50,6 %) | 61 / 85 (71,8 %) |
| Tổng tiền | 64 / 98 (65,3 %) | 71 / 98 (72,4 %) |
| Đúng cả 4 trường | 11 / 114 (9,6 %) | 21 / 114 (18,4 %) |
| F1 chịu lỗi OCR (macro) | 0.6636 | 0.7800 |
| Trường sai do OCR đọc sai chữ | 95 | 52 |

Khoảng tin cậy 95 % của hai cột còn chồng nhau; chưa có so sánh ghép cặp (việc P0.3).

Độ trễ (`eval/latency_v3_fixed.json`, Kaggle GPU T4, tập test, đo sau khi sửa lỗi cửa sổ): OCR 2,58 s mỗi hóa đơn
(tiền xử lý 0,11, CRAFT 0,62, VietOCR 1,85); khâu LayoutXLM 61,6 ms trên GPU (trung vị 57,3, p95 86,8, 60 hóa đơn) và
966,5 ms trên CPU (10 hóa đơn). Lần đo đầu cho 4586 ms trên GPU và 114 s trên CPU là **lỗi của chính code**: sau mỗi lần
gọi processor, tokenizer nhanh của transformers 5 còn giữ chế độ đệm 512 nên mỗi từ bị đếm 512 sub-token và mỗi hóa đơn bị
chia thành một cửa sổ cho mỗi từ (khoảng 127 lần forward thay vì 1). Lỗi chỉ xảy ra khi tạo cửa sổ từng hóa đơn một sau khi
đã gọi processor (đường đi của app và `measure_latency.py`), không xảy ra khi chấm theo lô, nên điểm đã chấm không đổi. Đã sửa
trong `word_token_counts` (commit `d33a3d6`, có test). Kiểm tra sau khi sửa: `verify_extractor.py` cho 115 / 115 hóa đơn
validation giống hệt dự đoán Tuần 4, 0 khác biệt, `PASS`.

## 2. Đầu ra mong muốn

```json
{
  "fields": {
    "TIMESTAMP":  {"text": "...", "date": "11/08/2020", "time": "21:19"},
    "TOTAL_COST": {"text": "...", "amount_vnd": 168000},
    "SELLER":     {"text": "MINIMART ANAN"},
    "ADDRESS":    {"text": "..."}
  },
  "items": [
    {"name": "Mì chính Ajinomoto 454g", "quantity": 1, "unit_price": 31000, "amount": 31000}
  ],
  "items_check": {"sum": 168000, "total": 168000, "status": "verified"}
}
```

- Trường nào hóa đơn không có (hoặc hệ thống không đọc được) thì **không xuất hiện** trong đầu ra gọn; mặt hàng không đọc
  được thì không có khóa `items`. Bản đầy đủ (từng từ OCR, tọa độ, thời gian) vẫn giữ cho gỡ lỗi.
- SELLER là một cột riêng, không bắt buộc: model vẫn trích, phần hiển thị quyết định có hiện hay không. Không huấn luyện lại
  để bỏ SELLER (sẽ làm mất hiệu lực con số test).
- `items_check.status`: `verified` khi tổng thành tiền các món bằng TOTAL_COST; `unverified` khi không khớp (app nên nhắc
  người dùng kiểm tra lại).

## 3. Cách làm và cách chấm (không cần gán nhãn tay)

- MC-OCR 2021 **không có nhãn mặt hàng** (chỉ 4 trường), và nhóm không gán nhãn tay được. Vì vậy module mặt hàng **không
  huấn luyện** và được chấm bằng **tổng tiền chuẩn có sẵn** (`TOTAL_COST` của MC-OCR): nếu các món đọc ra cộng lại đúng bằng
  tổng chuẩn thì gần như chắc chắn đã tìm đủ món và đọc đúng thành tiền từng món.
- Module `src/extraction/line_items.py` (thuần Python): ghép từ thành dòng theo tọa độ → khoanh bảng hàng (sau dòng tiêu đề
  cột, trước dòng "tổng") → mỗi dòng đọc số tiền ("36 .500", "24 000", "8, 000") và số lượng sao cho
  `số lượng × đơn giá = thành tiền` → tên lấy từ chính dòng đó hoặc dòng ngay trên (bố cục hai dòng) → dòng giảm giá (`KM`,
  số âm) trừ vào tổng. Luật viết bằng cách đọc chữ OCR của tập **train**; validation chỉ được chấm, không dùng để chỉnh.
- Chỉ số (`scripts/score_items.py`): hóa đơn đọc được ít nhất một món; **tổng các món = tổng chuẩn**; `verified` (tổng các món
  = số tiền in ở dòng tổng của chính hóa đơn) và **trong số `verified` bao nhiêu thật sự bằng tổng chuẩn** (cho biết app có
  tin được nhãn `verified` hay không); tỷ lệ dòng thỏa số lượng × đơn giá = thành tiền.
- **Không đo được độ đúng của tên hàng** (không có nhãn). Báo cáo chỉ đưa ví dụ ngẫu nhiên đọc bằng mắt, ghi rõ đó là minh họa.

### Kết quả (luật đóng băng ở phiên bản này, chạy trên token dataset `tok_ft_n05`)

| Chỉ số | Train (781 hóa đơn có tổng chuẩn) | Validation (94) | **Test (97, chấm một lần)** |
| --- | --- | --- | --- |
| Đọc được ít nhất một món | 85,4 % | 85,1 % | 80,4 % (78 / 97) |
| Tổng các món = tổng chuẩn | 56,9 % | 62,8 % [52,7; 71,9] | **47,4 %** (46 / 97) [37,8; 57,3] |
| Chỉ tính hóa đơn có từ 2 món trở lên | 60,0 % (257 / 428) | 72,0 % (36 / 50) | 46,6 % (27 / 58) |
| `verified` | 51,3 % | 55,3 % | 45,4 % (44 / 97) |
| Trong số `verified`, bằng tổng chuẩn | 97,0 % | 98,1 % (51 / 52) | 95,5 % (42 / 44) |
| Dòng thỏa số lượng × đơn giá = thành tiền | 91,5 % | 89,9 % | 91,9 % |

Lưu ý: validation ở đây chỉ có 94 hóa đơn vì chỉ tính hóa đơn có tổng chuẩn đọc được dưới 100 triệu. Một số hóa đơn có hàng
giảm giá nên tổng các món khác tổng cuối dù đọc đúng (đã trừ dòng giảm giá khi nhận ra, nhưng chưa phủ mọi kiểu). Tập test
chấm **một lần** (17/11/2026, `eval/items_test.json`) với luật không chỉnh thêm sau validation. Số test thấp hơn validation
(47,4 % so với 62,8 %); khoảng tin cậy của hai tập chồng nhau một phần, nên không kết luận chắc đâu là mức thật, và con số
test là con số đưa vào báo cáo. Nhãn `verified` vẫn đáng tin trên cả ba tập (95–98 % đúng).

## 4. Danh sách việc

Ưu tiên: P0 > P1 > P2. Mốc đã xong ghi ở cột "Trạng thái".

| # | Việc | Phụ thuộc | Xong khi | Trạng thái |
| --- | --- | --- | --- | --- |
| P0.1 | Tải về `test_results.zip` và `latency_v3.json` (cell 7 chỉ nén `eval/*test*`, nén riêng `latency_v3.json`) | — | Hai file nằm ngoài Kaggle | chờ |
| P0.2 | Tìm nguyên nhân khâu LayoutXLM chậm bất thường | P0.1 | Tìm ra (tokenizer giữ chế độ đệm), sửa, đo lại: 61,6 ms GPU, 966,5 ms CPU; `verify_extractor` PASS | xong |
| P0.3 | So sánh ghép cặp trên cùng 114 hóa đơn test giữa cấu hình cuối và M1b Tuần 4 | P0.1 | Khoảng tin cậy của hiệu số từng trường | chưa làm |
| 1 | Module đọc mặt hàng + tự kiểm tra + unit test | — | `tests/test_line_items.py` | xong |
| 2 | Chỉ số chấm bằng tổng chuẩn (`scripts/score_items.py`) | 1 | Bảng train, validation | xong |
| 3 | Đầu ra gọn (`compact_result`, `--items`, `--compact`): ẩn trường không đọc được, tách giờ khỏi ngày, SELLER riêng | 1 | `tests/test_extractor_items.py` | xong |
| 4 | Chấm mặt hàng trên test một lần (Python thuần, chạy trên Mac); unit test mới chạy lại ở Kaggle cell 3 | 1-3 | Bảng test | xong (test); cell 3 chờ |
| 5 | Ví dụ đọc bằng mắt: in 15 hóa đơn ngẫu nhiên (tên, số lượng, giá) kèm ảnh | 4 | Mục minh họa trong báo cáo | chưa làm |
| 6 | Nếu còn thời gian: nhãn tự sinh từ hóa đơn `verified` của train để huấn luyện model mặt hàng (M1c) | 4 | M1c không làm giảm 4 trường trên validation | P2 |
| 7 | README, báo cáo: mục mở rộng mặt hàng tách riêng khỏi kết quả 4 trường, ghi hạn chế | 4, P0.2 | Mọi con số có file nguồn | chưa làm |

## 5. Rủi ro

- OCR tên hàng có thể sai nhiều (VietOCR chỉ được tinh chỉnh trên dòng của 4 trường), ví dụ "Mi chính ahnomoto 4549". Giá ít
  rủi ro hơn vì chỉ là số.
- Luật không phủ mọi mẫu hóa đơn. Kiểm tra cộng dồn cho biết khi nào sai, không sửa được lỗi.
- Hóa đơn có giảm giá, VAT, phí dịch vụ làm tổng thành tiền ≠ TOTAL_COST dù đọc đúng → cần tính các dòng này vào kiểm tra.
- Gán nhãn là việc thủ công tốn thời gian nhất của kế hoạch.

## 6. Hạn chế đã biết

- Tên hàng chưa có số đo; chữ OCR của tên hàng có thể sai (VietOCR chỉ được tinh chỉnh trên dòng của 4 trường).
- Luật chưa phủ hóa đơn giảm giá nhiều tầng, hàng cân theo kg (số lượng thập phân) và bảng hàng không có dòng tiêu đề.
- Hóa đơn chụp nghiêng mạnh làm gộp dòng sai.
