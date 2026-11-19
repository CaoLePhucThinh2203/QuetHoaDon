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

So sánh ghép cặp trên cùng các hóa đơn (`scripts/compare_strict.py`, McNemar chính xác và bootstrap theo hóa đơn, file
`eval/paired_test.json`): SELLER +15,0 điểm [+7,5; +22,4] (p = 0,0004; 18 hóa đơn chỉ bản cuối đúng, 2 hóa đơn chỉ Tuần 4 đúng);
ADDRESS +11,3 [+5,7; +17,9] (p = 0,0005; 12 so với 0); ngày và giờ +17,0 [+9,4; +24,5] (p < 0,0001; 19 so với 1); cả bốn
trường +8,8 [+3,5; +14,0] (p = 0,002; 10 so với 0). Tổng tiền +7,1 [+1,0; +14,3] nhưng p = 0,065 (9 so với 2): chưa đủ để
kết luận tổng tiền cải thiện có ý nghĩa thống kê.

**Vòng 2 tinh chỉnh VietOCR: kết quả âm.** Tiếp tục huấn luyện từ trọng số vòng 1 (23 410 cặp, LR 5e-5, tối đa 20 epoch)
không giảm CER vùng validation: epoch 0 (điểm xuất phát) 6,28 %, rồi 6,98 / 6,56 / 6,82 / 6,73 %, dừng sớm sau 4 epoch không
cải thiện. Cổng đã chốt trước (khoảng tin cậy ghép cặp nằm dưới 0) không đạt nên M1b v4 không được huấn luyện; giữ cấu hình
VietOCR vòng 1 + M1b v3. Hàm mất mát đứng yên khoảng 0,50 ngay từ epoch 1, cho thấy mô hình đã bão hòa trên dữ liệu này.

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
| P0.1 | Tải về `test_results.zip` | — | File nằm ngoài Kaggle và trên Drive | xong |
| P0.2 | Tìm nguyên nhân khâu LayoutXLM chậm bất thường | P0.1 | Tìm ra (tokenizer giữ chế độ đệm), sửa, đo lại: 61,6 ms GPU, 966,5 ms CPU; `verify_extractor` PASS | xong |
| P0.3 | So sánh ghép cặp trên cùng 114 hóa đơn test giữa cấu hình cuối và M1b Tuần 4 | P0.1 | Khoảng tin cậy của hiệu số từng trường | xong |
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
- Luật chưa phủ hóa đơn giảm giá nhiều tầng. Hàng cân theo kg và bảng có dòng tiêu đề bị OCR đọc sai: bản luật 3 (mục 8).
- Hóa đơn có giảm giá trên tổng đơn ("Tổng 78 000 → chiết khấu → 70 200"): các món có thể đọc đúng nhưng vẫn bị tính
  sai, vì tổng chuẩn là số sau giảm.
- Hóa đơn chụp nghiêng làm gộp dòng sai: bản luật 2 (mục 7) sửa phần lớn; giấy bị cong (độ nghiêng khác nhau trên cùng
  một tờ) vẫn có thể gộp sai.

## 7. Bản luật 2 (18/11/2026): gom dòng theo độ nghiêng, gộp số bị tách

**Vì sao.** Đọc chữ OCR của tập **train** thấy hai lỗi lặp lại ở bước ghép từ thành dòng:

1. Ảnh chụp nghiêng: dòng được gom theo tọa độ y nằm ngang, nên đầu bên phải của một dòng in bị xếp vào dòng khác, tên
   hàng bị cắt đôi và số tiền tách khỏi tên. Trên train, 62 hóa đơn có độ nghiêng ≥ 0,02 chỉ đúng tổng 37 % so với 59 % ở
   ảnh thẳng.
2. Số bị tách: CRAFT cắt một số thành hai hộp chồng nhau và VietOCR đọc ký tự ở chỗ chồng hai lần, ví dụ `399` + `9,000`
   (của 399,000) hay `6,` + `,250` (của 6,250). Luật cũ đọc thành hai số khác nhau.

**Thay đổi** (`src/extraction/line_items.py`, chỉ phần mặt hàng; 4 trường không đổi):

- `text_slope`: tìm độ nghiêng bằng hình chiếu (thử -0,2 đến 0,2, bước 0,005; chọn độ nghiêng xếp các tâm chữ thành ít
  hàng đầy nhất) và chỉ dùng khi tốt hơn nằm ngang ít nhất 10 %, nên ảnh thẳng được gom y như cũ. Cách lấy trung vị độ dốc
  của từng cặp từ kề nhau đã thử trước và bị bỏ vì quá nhiễu (12–30 cặp mỗi hóa đơn; làm hỏng 3 hóa đơn train).
- `merge_split_numbers`: gộp hai hộp số kề nhau khi hai hộp chồng nhau và ký tự cuối hộp trái trùng ký tự đầu hộp phải,
  với điều kiện ký tự đó là dấu phân cách hoặc đứng ngay trước dấu phân cách. Điều kiện sau thêm vào sau khi bản đầu gộp
  nhầm `10` + `000` (số "10 000" viết có dấu cách) thành 1000.
- Dòng có cụm "giá đăng ký / giá niêm yết / giá gốc" ở giữa (tên hàng bị nối vào khi đã chỉnh nghiêng): phần trước cụm
  được giữ làm tên, dòng bị bỏ như trước.

**Cách chọn.** Luật viết và chỉnh chỉ trên **train**; validation chấm một lần sau khi đã chốt; **test không chấm lại**
(con số test 47,4 % ở mục 3 vẫn là con số của bản luật 1, bản đã đóng băng cùng cấu hình cuối).

| Chỉ số (`tok_ft_n05`) | Train, bản 1 | Train, bản 2 | Validation, bản 1 | Validation, bản 2 |
| --- | --- | --- | --- | --- |
| Tổng các món = tổng chuẩn | 444 / 781 (56,9 %) | **486 / 781 (62,2 %)** [58,8; 65,6] | 59 / 94 (62,8 %) | **64 / 94 (68,1 %)** [58,1; 76,6] |
| Hóa đơn sửa được / bị làm hỏng | | 43 / 1 (McNemar p < 0,0001) | | 5 / 0 (p = 0,0625) |
| `verified` | 401 (51,3 %) | 477 (61,1 %) | 52 (55,3 %) | 59 (62,8 %) |
| Trong số `verified`, bằng tổng chuẩn | 97,0 % | 97,7 % | 98,1 % | 98,3 % |
| Dòng thỏa số lượng × đơn giá = thành tiền | 91,5 % | 91,9 % | 89,9 % | 90,9 % |

Từng phần trên train (sửa / làm hỏng so với bản 1): chỉ chỉnh nghiêng 6 / 0; chỉ gộp số 35 / 0; cả hai 43 / 1. Hóa đơn
bị làm hỏng (`...014uqaoc`) có giấy cong: sau khi chỉnh nghiêng, thành tiền 52,000 lệch xuống dòng riêng, mất tên. Trên
validation cả 5 hóa đơn được sửa là nhờ gộp số; chỉnh nghiêng không đổi kết quả hóa đơn nào ở đây.

Đọc kết quả: cải thiện trên train rõ ràng; trên validation cùng chiều (5 sửa, 0 hỏng) nhưng 94 hóa đơn chưa đủ để có ý
nghĩa thống kê. Tên hàng vẫn không đo được; chỉnh nghiêng giữ tên đủ hơn ở các ví dụ đã xem, đó chỉ là minh họa.

**Ảnh hưởng.** Cell 7 của `notebooks/07_final_evaluation_kaggle.ipynb` kiểm tra số test của bản 1 (46 / 97) nên sẽ in LỆCH
khi chạy với code mới; số đó và báo cáo cuối vẫn đúng cho cấu hình đã đóng băng. Web demo và `extract_receipt.py --items`
dùng bản 2.

File nguồn: `eval_items_v2/items_train_v2.json`, `eval_items_v2/items_validation_v2.json` (chạy
`scripts/score_items.py --token-dir phase1_eval/tok_ft_n05`, Python thuần trên máy cá nhân).

## 8. Bản luật 3 (19/11/2026): hàng cân, tiêu đề bị đọc sai, dòng tổng giả

Ưu tiên của nhóm từ đây: **tổng tiền, tên hàng, đơn giá và số lượng**; người bán, địa chỉ, ngày giờ không cần cải thiện thêm.

**Vì sao.** Phân loại 295 hóa đơn train còn sai sau bản 2: không đọc được món nào 112 (phần lớn là phiếu thanh toán thẻ,
biên nhận nạp tiền, báo cáo cuối ca: không có mặt hàng nên đọc ra rỗng là đúng); có dòng sai phép tính 89; thiếu món 54;
thừa dòng hoặc giảm giá trên tổng đơn 40. Đọc các hóa đơn train trong từng nhóm thấy bốn lỗi của luật:

1. **Hàng cân theo kg** (VinMart, Co.op): `Rau ngót 40 000/KG X 0,322 KG` rồi dòng mã vạch `12 .880 12.880`. Luật cũ đọc cân
   nặng "0,322" thành số tiền 322 đồng và bỏ mất dòng tiền thật vì không còn tên.
2. **Dòng tiêu đề bị OCR đọc sai** ("S1. Đơn giú Thành tiền") chỉ khớp một từ khóa, luật cần hai.
3. **Dòng "tổng" giả ở đầu hóa đơn**: VietOCR tinh chỉnh in "Tổng thanh toán" ở vùng không đọc được, luật tưởng bảng đã kết
   thúc ngay dòng đầu. "Tổng số lượng hàng" (Co.op) cũng bị nhận làm tiêu đề.
4. **"giò heo"** bỏ dấu thành "gio", trùng với "giờ" (dòng ngày giờ), nên tên hàng bị bỏ.

**Thay đổi** (`src/extraction/line_items.py`):

- `_per_kg`: dòng có "/KG" là dòng tên; giữ đơn giá theo kg và cân nặng (cả khi cân nặng nằm một dòng riêng), lấy thành
  tiền ở dòng sau. Khi đơn giá × cân nặng = thành tiền (sai số ≤ 0,5 % hoặc 2 đồng), món có số lượng thập phân (0.322) và
  đơn giá theo kg.
- Tiêu đề: thêm một cụm mạnh ("thành tiền", "đơn giá", "tên hàng", "mặt hàng", "số lượng") trên dòng không có số tiền;
  dòng có từ "tổng" không bao giờ là tiêu đề.
- Dòng "tổng" mà trước nó không có dòng số tiền nào thì không phải cuối bảng; tìm dòng tổng kế tiếp. Không có dòng tổng nào
  sau đó thì giữ dòng đầu như cũ (phiếu không có bảng hàng).
- "gio" không còn chặn tên hàng (vẫn dùng để tìm đầu bảng).

**Cách chọn.** Như bản 2: viết và chỉnh trên **train**, validation chấm một lần sau khi chốt, **test không chấm lại**.

| Tổng các món = tổng chuẩn | Bản 1 (`main`) | Bản 2 | **Bản 3** |
| --- | --- | --- | --- |
| Train (781) | 444 (56,9 %) | 486 (62,2 %) | **554 (70,9 %)** [67,7; 74,0] |
| Validation (94) | 59 (62,8 %) | 64 (68,1 %) | **69 (73,4 %)** [63,7; 81,3] |

| So sánh ghép cặp | Sửa được / làm hỏng | McNemar p |
| --- | --- | --- |
| Train, bản 2 → 3 | 68 / 0 | < 0,0001 |
| Validation, bản 2 → 3 | 5 / 0 | 0,0625 |
| Validation, bản 1 → 3 | 10 / 0 | 0,002 |

| Chỉ số khác (validation) | Bản 1 | Bản 3 |
| --- | --- | --- |
| Đọc được ít nhất một món | 85,1 % | 90,4 % |
| `verified` | 55,3 % | 68,1 % |
| Trong số `verified`, bằng tổng chuẩn | 98,1 % | 98,4 % (63 / 64) |
| Dòng thỏa số lượng × đơn giá = thành tiền | 89,9 % | 91,2 % |

Trên train, độ chính xác của `verified` giữ ở 97,1 % (531 / 547; bản 1: 97,0 %). Một bản trung gian tăng điểm nhưng làm
"tìm thấy dòng tổng" trên validation giảm từ 97,9 % xuống 89,4 % (bỏ qua dòng tổng của phiếu không có bảng hàng); đã sửa
bằng cách giữ dòng tổng đầu khi không có dòng nào sau nó, trước khi chốt.

**Còn lại trên train (227 hóa đơn).** Thiếu món 66; dòng sai phép tính 55 (phần lớn chữ số OCR đọc sai, như
`11452 .500 .500`); thừa hoặc giảm giá trên tổng đơn 43; 32 ảnh mà VietOCR tinh chỉnh "bịa chữ" (cụm "Tổng thanh toán",
"104 Trần Phú" lặp nhiều lần ở ảnh mà VietOCR gốc cũng không đọc được: 53 / 922 train, 5 / 115 validation, 7 / 116 test,
không gặp ở VietOCR gốc); 31 không đọc được món nào. Hai nhóm đầu cần OCR tốt hơn; nhóm ảnh bịa chữ cần một bước nhận ra ảnh
không đọc được để app báo cho người dùng thay vì đưa ra số tiền bịa.

File nguồn: `eval_items_v3/items_train_v3.json`, `eval_items_v3/items_validation_v3.json` (cùng lệnh như bản 2).

### 8b. Số lượng thập phân không có "/KG" (19/11/2026)

Thử web demo trên một hóa đơn SunMart thấy `Thịt bò thăn` ra SL 52 thay vì 0,52 kg. Trên train có cùng kiểu ở 2 hóa đơn
(9 dòng): `0.39 Kg 25,000 9,750` (`...013iulwh`), `0.146 155,000 22,630` (`...014xyqra`). Luật cũ hoặc bỏ qua số thập phân
(SL 1, dòng sai phép tính), hoặc đọc ngược: "0.146" thành 146 đồng và SL 155, vẫn thỏa 155 × 146 = 22 630 vì cân nặng ×
giá/kg luôn có cùng chữ số với (giá ÷ 1000) × (cân nặng × 1000).

Sửa (`_decimal_quantity`): thử trước một số thập phân 0 < q < 100 làm số lượng, chỉ nhận khi q × đơn giá = thành tiền
(sai số ≤ 0,5 % hoặc 2 đồng) với hai số tiền khác trên dòng. Tổng các món không đổi (thành tiền vốn đã đúng); đổi 9 dòng
train và 1 dòng validation, cả 10 đều đúng hơn khi đọc lại (ví dụ `BA RỌI HEO` 180 × 248 → 0,248 kg × 180 000); dòng thỏa
phép tính: train 1864 → 1868, validation 206 → 207 trên 226.

**Vì sao.** Trên train, khi phần mặt hàng (bản 3) báo `verified`, tổng các món bằng tổng chuẩn ở **530 / 547 hóa đơn
(96,9 %)**; trong khi model chỉ đọc đúng tổng tiền ở 76,5 % hóa đơn validation. Trong 17 ca `verified` sai trên train, 3 ca
là giảm giá trên tổng đơn và tổng chuẩn vẫn là một số in trên dòng "tổng".

**Luật** (`src/extraction/total_check.py`, không có tham số để chỉnh; suy ra từ số liệu train ở trên):

- Các món `verified` **và** số tổng của model **không** in trên dòng "tổng" nào (hoặc model không ra số) → dùng tổng các món.
- Số của model có in trên một dòng "tổng" → giữ nguyên (giữ được tổng sau giảm giá trên tổng đơn).

Bật bằng `--total-from-items` (`decode_from_probs.py`, `extract_receipt.py`, `demo_app.py`) hoặc
`ReceiptExtractor(total_from_items=True)`. **Mặc định tắt**: cấu hình đã đóng băng và mọi số trong báo cáo cuối không đổi.

**Kết quả, validation, chấm một lần** (M1b v3 + luật ngày giờ + làm sạch, xác suất đã lưu; **test không chấm lại**):

| | Cấu hình cuối | + tổng từ các món |
| --- | --- | --- |
| Tổng tiền đúng | 75 / 98 (76,5 %) | **80 / 98 (81,6 %)** [72,8; 88,1] |
| Không ra tổng tiền | 3 | 1 |
| Đúng cả 4 trường | 20 / 115 | 21 / 115 |

Đổi 8 hóa đơn (6 có tổng chuẩn): sửa đúng 5, làm hỏng 0 (McNemar p = 0,0625); 1 hóa đơn trước không ra tổng nay ra một
số sai (105 000 so với 405 000): vẫn sai, nhưng app sẽ hiện một số sai thay vì để trống. Các ca được sửa: model chọn tiền
khách đưa (150 000 → 148 000), chọn một món (3 000 → 322 800), hoặc không ra số.

File nguồn: `eval_total_v1/decoded_v3_val_final.json`, `eval_total_v1/decoded_v3_val_total_items.json`
(`decode_from_probs.py --probs eval/m1b_v3_val_probs.npz --datetime --cleanup [--total-from-items]`).
