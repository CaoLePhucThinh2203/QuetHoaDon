# Rà lỗi mặt hàng có hình: 300 hóa đơn train (23/11/2026)

**Cách làm.** `scripts/make_review_cases.py` tạo 300 ca từ tập **train** (209 hóa đơn mà tổng các món ≠ tổng chuẩn với
luật hiện tại, cộng 91 hóa đơn đúng làm đối chứng): ảnh gốc MC-OCR, chữ OCR theo dòng, các món luật đọc ra, kết quả đối
chiếu. 10 subagent Claude Sonnet (`.claude/agents/receipt-reviewer.md`) chạy song song, mỗi agent 30 ca: xem ảnh, so
với OCR và luật, ghi nguyên nhân chính, chất lượng tên hàng và ý tưởng luật chung. Không đọc validation / test cho việc
này. Kết quả gốc: `kaggle/review/out/chunk_*.json` (ngoài Git).

**Độ tin cậy.** Khớp với phân loại tự động ở chỗ kiểm được: 29 / 29 ảnh bị gắn cờ "khó đọc" được agent xếp
`unreadable_photo`; 24 / 27 hóa đơn "không có bảng hàng" được xếp `not_itemized`; 75 / 91 ca đối chứng xếp `correct`
(16 còn lại: thành tiền đúng nhưng tên hoặc số lượng sai). Kiểm tay `...014cdjax`: đúng như agent nói, luật lấy chữ
"VAT10+" làm tên hàng.

## Nguyên nhân (209 hóa đơn sai)

| Nhóm | Nguyên nhân | Số hóa đơn | Tỉ lệ |
| --- | --- | --- | --- |
| Không phải lỗi đọc hàng | Phiếu không có bảng hàng (thẻ, nạp tiền, báo cáo ca) | 43 | 21 % |
| | Ảnh khó đọc (mờ, xoay 90°, cắt) | 33 | 16 % |
| | Giảm giá / VAT / phí trên tổng đơn (các món đúng) | 17 | 8 % |
| | Nhãn tổng tiền lỗi | 8 | 4 % |
| OCR | Đọc sai chữ số | 23 | 11 % |
| | Bỏ sót chữ / dòng | 15 | 7 % |
| **Luật (sửa được)** | Gán số lượng / đơn giá / thành tiền sai | 25 | 12 % |
| | Lấy tên sai dòng hoặc cắt tên | 22 | 11 % |
| | Sai đầu / cuối bảng hàng | 17 | 8 % |
| | Dòng không phải món bị đọc thành món | 4 | 2 % |
| | (agent cho là đúng) | 2 | 1 % |

Khoảng **một nửa (48 %)** hóa đơn "sai" thực ra không có lỗi đọc mặt hàng hoặc không thể sửa bằng luật; **18 %** do OCR;
**33 % (68 hóa đơn) do luật**, là phần còn cải thiện được.

**Tên hàng** (209 hóa đơn sai): tốt 21, một phần (lỗi chính tả OCR, bị cắt) 87, sai 52, không áp dụng 49. Đối chứng
(91): tốt 47, một phần 41, sai 3.

## Các mẫu lỗi luật lặp lại (từ gợi ý của agent, đếm thô)

1. **Co.op / Saigon Co.op:** mỗi món hai dòng "mã vạch + tên" rồi "VAT10% đơn giá thành tiền" (có khi thêm dòng
   "x.xxx KG"): luật lấy "VAT10+" làm tên, đọc chữ số của "VAT05%" thành số lượng (khoảng 15 hóa đơn).
2. **Từ đơn vị (CHA, HOP, GOI, CAI, TUI…) bị lấy làm tên** ở dòng số lượng của Minimart ANAN / VinMart; tên thật ở dòng
   mã vạch phía trên (nhiều ca).
3. **Chữ số của mã vạch hoặc của kích cỡ trong tên** ("500g", "100ml", "PL101") bị đọc thành số lượng / đơn giá
   (khoảng 15 hóa đơn mỗi loại).
4. **Từ khóa "Tổng…" lẫn trong dòng món** (OCR ghép dòng) làm bảng kết thúc sớm (khoảng 12 hóa đơn).
5. Dòng "Giá đăng ký / Giá bán", bố cục không tiêu đề hoặc tiêu đề tiếng Anh, dòng KM / trả hàng (ít hơn).

Việc tiếp: sửa các mẫu 1–4 theo thứ tự, mỗi mẫu đo trên train (sửa / hỏng) rồi chấm validation một lần.
