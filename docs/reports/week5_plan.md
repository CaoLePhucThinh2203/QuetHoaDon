# Kế hoạch Tuần 5 — Final Integration & Reporting

*Soạn ngày 17/11/2026. Đây là tuần cuối; nhóm đặt hạn hoàn thành sản phẩm trong ngày 17/11/2026. File này là kế hoạch hiện hành;
[`project_plan.md`](project_plan.md) là kế hoạch tổng soạn ngày 10/11 và chỉ còn dùng để tra mã task.*

## 1. Yêu cầu Tuần 5

| # | Yêu cầu của giảng viên |
| --- | --- |
| W5-1 | Full deep learning pipeline demonstration: Data → Model → Output |
| W5-2 | Final model performance summary |
| W5-3 | Limitations & risks |
| W5-4 | Future improvements |
| W5-5 | Final GitHub review: Code, Documentation, Report, Reproducibility |

Outcome: một giải pháp AI đầu-cuối đã hoàn thiện, có tài liệu đầy đủ. Rubric cuối kỳ chấm 5/10 điểm cho việc dùng deep
learning trả lời 3 câu hỏi nghiên cứu, và 1 điểm cho repo (code, tài liệu, tiền xử lý, notebook hoặc script, hướng dẫn chạy).

## 2. Hiện trạng đã kiểm chứng (17/11/2026)

Mỗi dòng ghi rõ nguồn. Không có số nào được ước lượng.

| Hạng mục | Hiện trạng | Nguồn |
| --- | --- | --- |
| Nhánh `main` | Commit `69d4775` (PR #10), CI xanh; 10 PR đã merge, không có PR mở, không có Issue | GitHub API |
| Commit chưa merge | `ed731c3` trên `HoaiNam/m1b-training` (xóa file báo cáo Tuần 4 và tên ứng dụng cũ) + phần dọn tài liệu trên `HoaiNam/final-cleanup` | `git log` |
| Mã nguồn | Đủ 5 chặng: tiền xử lý, CRAFT, VietOCR, LayoutXLM, trích trường; có huấn luyện, baseline B0, chấm điểm, so sánh, phân tích lỗi, tìm ảnh trùng | `src/`, `scripts/` |
| Unit test | 157 hàm test trong 16 file. Lần chạy đầy đủ gần nhất là trên Kaggle ngày 11/11; hôm nay chưa chạy lại | đếm trong `tests/` |
| Trọng số | M1b v2 còn (dataset Kaggle `m1b-v2-result` và file zip trên máy). M1a **đã mất checkpoint**, còn `run.json`. B0 còn file `b0_k32.pkl` | `week4_eval.zip`, ghi nhận của nhóm |
| Notebook | 01, 02, 03 có trong repo nhưng **không có output nào được lưu** | đọc file `.ipynb` |
| Demo đầu-cuối | **Chưa có.** `scripts/run_pipeline.py` chỉ chạy tới OCR, chưa có hàm ảnh → 4 trường | `scripts/`, `src/pipeline/` |

### Kết quả Tuần 4 (đã đối chiếu với file gốc)

Nguồn: `week4_eval.zip` → `eval/comparison_val.md`, `eval/comparison_test.md`, `eval/*_test.json`. F1 khớp chịu lỗi OCR, giải mã mềm +
luật; ngoặc vuông là khoảng tin cậy bootstrap 95%; một seed (42).

| Tập | Model | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | Macro F1 | Đúng cả hóa đơn | Tổng tiền đúng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validation (115) | B0 | 0.000 | 0.108 | 0.732 | 0.519 | 0.340 [0.303, 0.375] | 1 / 115 | 55 / 98 |
| | M1a | 0.654 | 0.459 | 0.753 | 0.648 | 0.628 [0.568, 0.688] | 32 / 115 | 67 / 98 |
| | M1b | 0.688 | 0.500 | 0.759 | 0.689 | 0.659 [0.602, 0.716] | 34 / 115 | 71 / 98 |
| Test (116) | B0 | 0.000 | 0.065 | 0.788 | 0.553 | 0.352 [0.317, 0.386] | 0 / 116 | 60 / 98 |
| | M1a | 0.685 | 0.469 | 0.794 | 0.604 | 0.638 [0.582, 0.691] | 30 / 116 | 64 / 98 |
| | M1b | 0.700 | 0.502 | 0.794 | 0.634 | 0.658 [0.600, 0.709] | 32 / 116 | 64 / 98 |

Các số khác đã có file kết quả:

- Trần do OCR và ghép nhãn (giải mã từ nhãn chuẩn, test): SELLER 0.864, ADDRESS 0.616, TIMESTAMP 0.727, TOTAL_COST 0.786.
- Nguyên nhân 159 trường sai của M1b trên test (phân loại tự động): OCR đọc sai chữ 99, model chọn sai 41, nhãn tổng tiền
  không có số 10, model bỏ sót 7, không từ OCR nào mang nhãn 2. Phía OCR và nhãn chiếm 111 / 159 (70%).
- Tổng tiền của M1b trên test: đúng 64 / 98, không dự đoán được số 13; MAE 198.763 VND, RMSE 1.088.519 VND, trung vị sai số 0.
- ROC-AUC macro theo token trên test: B0 0.839, M1a 0.894, M1b 0.841.
- Độ trễ chặng LayoutXLM trên GPU T4 (đã gồm đọc ảnh): M1a 32 ms, M1b 33 ms mỗi hóa đơn. B0 chưa đo.
- B0: k = 32, Silhouette 0.088 (`eval/b0_val.json`).
- M1a: epoch tốt nhất 16, 571 giây, 6.921 tham số huấn luyện (`runs/m1a_*/run.json`). M1b v2: epoch tốt nhất 15, 29,4 phút
  (theo `docs/training.md`; `run.json` nằm trong file zip trọng số, hôm nay chưa mở lại).
- Ảnh gần trùng: README ghi M1b còn macro F1 0.642 trên test khi loại hóa đơn có bản gần trùng. **File kết quả của bước
  này không nằm trong `week4_eval.zip`**, nên cần chạy lại để lưu bằng chứng (task C1).

## 3. Khoảng trống so với yêu cầu

| Yêu cầu | Đã có | Còn thiếu |
| --- | --- | --- |
| W5-1 Demo | Từng chặng chạy riêng được; CLI một ảnh tới OCR | Hàm và CLI ảnh → 4 trường dùng M1b; notebook demo có output; (tùy chọn) giao diện Gradio và video |
| W5-2 Tổng kết hiệu năng | Bảng B0 · M1a · M1b trên validation và test | Tài liệu tổng kết trong repo; độ trễ trên CPU và của B0; kết quả cho RQ1; phần phân loại chi tiêu; độ dao động giữa các seed |
| W5-3 Hạn chế và rủi ro | Nội dung nằm rải rác trong báo cáo Tuần 4 (PDF) | Một mục viết trong repo |
| W5-4 Hướng cải tiến | Như trên | Một mục viết trong repo |
| W5-5 Review GitHub | README đã có kết quả, cách chạy, phiên bản đã dùng | Merge commit còn treo; notebook có output; sửa tài liệu cũ (mục E); quyết định cách chia sẻ trọng số |

Tình trạng ba câu hỏi nghiên cứu:

| RQ | Tình trạng | Ghi chú |
| --- | --- | --- |
| RQ1 tiền xử lý và augmentation | **Chưa có thí nghiệm** | Không có script đo CER theo cấu hình tiền xử lý; chưa fine-tune VietOCR |
| RQ2 trích tổng tiền | Đã trả lời | So sánh B0, M1a, M1b |
| RQ2 phân loại chi tiêu | **Chưa làm** | Chưa có nhãn danh mục, chưa có code |
| RQ3 độ chính xác và độ trễ | Một phần | Có thời gian từng chặng trên GPU; chưa đo LayoutXLM trên CPU, chưa đo B0 |

## 4. Nguyên tắc cho Tuần 5

1. Cấu hình cuối đã chốt ở Tuần 4: M1b v2 + giải mã mềm + luật. Tập test đã chấm; **không dùng test để chọn hay chỉnh bất cứ thứ gì**.
   Thí nghiệm mới chỉ chạy trên validation. Thêm seed chỉ để báo độ dao động của cấu hình đã chốt.
2. Báo cáo chỉ ghi số có file kết quả đi kèm. Mục nào chưa đo thì ghi "chưa đo".
3. Mọi lần chạy model và test thực hiện trên Kaggle; máy cá nhân chỉ sửa code và quản lý git.
4. Không làm ứng dụng di động. Demo là notebook (và Gradio nếu kịp).

## 5. Danh sách task

Ưu tiên: **P0** bắt buộc xong trong ngày; **P1** làm nếu kịp, không kịp thì ghi vào mục hạn chế; **P2** không làm trong Tuần 5,
đưa vào hướng cải tiến.

### A. Hợp nhất repo

| # | Việc | Ưu tiên | Ở đâu | Xong khi |
| --- | --- | --- | --- | --- |
| A1 | Mở PR `HoaiNam/final-cleanup` → `main`, chọn Squash and merge, message không có dấu hai chấm | P0 | GitHub (web) | `main` không còn thư mục `reports/`, không còn tên ứng dụng cũ |
| A2 | Sửa notebook 03 (cell 2) và `docs/training.md` để lấy code từ `main` thay cho nhánh `HoaiNam/m1b-training` | P0 | máy | Notebook chạy được sau khi xóa nhánh |
| A3 | Xóa nhánh `HoaiNam/m1b-training` và `HoaiNam/final-cleanup` sau khi merge; đồng bộ `main` trên máy | P0 | git | Chỉ còn `main` và nhánh đang làm |

### B. Demo đầu-cuối (W5-1)

| # | Việc | Ưu tiên | Ở đâu | Xong khi |
| --- | --- | --- | --- | --- |
| B1 | `src/pipeline/`: lớp trích xuất đầu-cuối ảnh → OCR → từ và toạ độ → LayoutXLM → giải mã → 4 trường, dùng **đúng hàm giải mã đã chấm điểm** (`decode_predictions_soft` + `fill_missing`). Nhận được cả đầu vào là kết quả OCR có sẵn, phòng khi CRAFT/VietOCR không cài được trên Kaggle | P0 | máy (viết), Kaggle (chạy) | Unit test cho phần thuần Python qua; chạy được một ảnh |
| B2 | `scripts/extract_receipt.py`: `--image`, `--checkpoint`, `--output`, in JSON 4 trường kèm thời gian từng chặng | P0 | máy, Kaggle | Một lệnh cho ra JSON |
| B3 | `notebooks/04_demo_end_to_end.ipynb`: Data → Model → Output trên ảnh mẫu và vài hóa đơn validation; hiện ảnh, ô chữ, JSON, thời gian. Lưu notebook kèm output rồi đưa vào repo | P0 | Kaggle | Notebook có output trong repo |
| B4 | Giao diện Gradio trong notebook và video quay màn hình dự phòng | P1 | Kaggle, nhóm | Có link hoặc video |

Lưu ý: CRAFT và VietOCR mới được kiểm chứng trên Colab (lần chạy OCR ngày 10/11), chưa kiểm chứng trên Kaggle.

### C. Bổ sung bằng chứng cho hiệu năng và 3 RQ (W5-2)

| # | Việc | Ưu tiên | Ở đâu | Xong khi |
| --- | --- | --- | --- | --- |
| C1 | Chạy lại `find_duplicates.py` và `rescore_without_duplicates.py`, lưu file kết quả | P0 | Kaggle (CPU) | Có file JSON hoặc Markdown trong gói kết quả |
| C2 | RQ3: script đo độ trễ chặng LayoutXLM trên GPU và CPU, độ trễ của B0, kích thước model; ghép với thời gian CRAFT và VietOCR đã có → bảng độ chính xác so với độ trễ | P0 | máy (viết), Kaggle | `latency.json` và bảng |
| C3 | RQ1 (tiền xử lý): đo CER và WER của VietOCR trên các vùng có nhãn của tập validation với 4 cấu hình (không xử lý, deskew, tăng tương phản, đầy đủ), tách theo nhóm `anno_image_quality` | P0 | máy (viết), Kaggle | Bảng CER / WER |
| C4 | RQ1 (độ bền): thêm ảnh làm mờ, làm tối, thêm nhiễu nhân tạo vào cùng thí nghiệm C3 | P1 | Kaggle | Bảng độ bền |
| C5 | RQ1 (augmentation): fine-tune VietOCR trên vùng có nhãn của tập train, có và không augmentation, so CER trên validation. Thử một lần; nếu bộ huấn luyện của VietOCR không chạy được trong môi trường hiện tại thì dừng và ghi vào hạn chế | P1 | Kaggle | Hai giá trị CER, hoặc ghi chú lý do không chạy được |
| C6 | RQ2 (phân loại chi tiêu): gán tay danh mục cho 231 hóa đơn validation và test; nhãn train sinh từ từ khóa; so TF-IDF + LR với XLM-R fine-tune bằng macro F1 trên phần gán tay | P1 | nhóm (gán nhãn), Kaggle | Bảng macro F1 và file nhãn |
| C7 | Thêm seed: M1b seed 1 và 2; M1a seed 42 (train lại vì mất checkpoint), 1 và 2; tổng hợp bằng `aggregate_seeds.py` | P1 | Kaggle (M1a khoảng 10 phút, M1b khoảng 30 phút mỗi lần) | Trung bình ± độ lệch chuẩn |
| C8 | Vẽ 30 hóa đơn M1b sai bằng `draw_errors.py` rồi xem tay, đối chiếu với phân loại lỗi tự động | P1 | Kaggle, nhóm | Ghi lại số ca khớp và không khớp |
| C9 | Đánh giá định tính ITEM_NAME, ITEM_PRICE, QUANTITY | P2 | — | Đưa vào hướng cải tiến |

### D. Báo cáo cuối trong repo (W5-2, W5-3, W5-4)

| # | Việc | Ưu tiên | Ở đâu | Xong khi |
| --- | --- | --- | --- | --- |
| D1 | `docs/reports/week5_final_report.md`: tổng kết hiệu năng (bảng ở mục 2 và kết quả nhóm C), trả lời từng RQ kèm tình trạng bằng chứng, hạn chế và rủi ro, hướng cải tiến, cách tái lập | P0 | máy | Mọi số trong file đều truy được về một file kết quả |
| D2 | Cập nhật README: dòng trạng thái, bảng kết quả, liên kết tới báo cáo cuối và notebook demo | P0 | máy | README khớp với báo cáo |

### E. Review GitHub cuối (W5-5)

| # | Việc | Ưu tiên | Xong khi |
| --- | --- | --- | --- |
| E1 | **Code:** CI xanh trên `main` sau khi merge; unit test chạy đạt trên Kaggle; không có khóa bí mật hay dữ liệu trong repo | P0 | Ảnh chụp hoặc log kèm theo |
| E2 | **Documentation:** `docs/architecture.md` còn dòng "Stage 4 not implemented"; `docs/dataset.md` còn số liệu cũ (vùng theo split 5308 / 643 / 634, "1 rỗng / 1 lỗi") và mục Known issues đã lỗi thời (1044 so với 1153 ảnh, "chưa có tìm ảnh trùng") | P0 | Tài liệu khớp với kết quả đã kiểm chứng |
| E3 | **Report:** báo cáo cuối D1 trong repo; bản docx hoặc PDF do nhóm hoàn thiện (mục F) | P0 | Có cả hai |
| E4 | **Reproducibility:** phiên bản đã dùng (README 5.4), seed, lệnh (`docs/training.md`), notebook 01, 03, 04 có output, nguồn dữ liệu, cách lấy trọng số | P0 | Người ngoài làm theo README chạy lại được |
| E5 | Quyết định cách chia sẻ trọng số M1b: giữ dataset Kaggle private và ghi cách train lại, hoặc công khai | P0 | Ghi trong README |
| E6 | Nhờ chủ nhánh xóa 3 nhánh đã merge không thuộc nhóm task này (`chore/restructure-repo`, `docs/update-readme`, `update-week-3-and-4`) | P1 | Repo chỉ còn `main` |

### F. Tài liệu ngoài repo (nhóm thực hiện)

| # | Việc | Ưu tiên |
| --- | --- | --- |
| F1 | Sửa trang 11 của `AI Project Progress`: đoạn JSON Tuần 3 vẫn ghi `label_distribution` chỉ có ADDRESS 1880 và SELLER 1153; số đúng nằm ở Phụ lục A cùng file | P0 |
| F2 | Cập nhật mục 4 và bảng rủi ro của báo cáo Tuần 4: kiểm tra ảnh gần trùng và `docs/training.md` đã làm | P0 |
| F3 | Điền các chỗ `[tên]`, `[ngày]`, `[điền]` (phân công, mốc, số giờ) | P0 |
| F4 | Viết phần Tuần 5 trong docx, slide, và quay video demo | P0 |
| F5 | Ký trang cam kết chống đạo văn và gửi email nộp bài | P0 |

## 6. Thứ tự thực hiện

1. A1 (merge) để `main` sạch.
2. Viết code B1, B2, C2, C3 kèm unit test và sửa A2 trên một nhánh `HoaiNam/...` mới; đẩy lên GitHub.
3. Trên Kaggle: chạy unit test, rồi C1, C2, C3, B3. Tải gói kết quả về.
4. Song song trên Kaggle nếu còn thời gian: C7 (tốn GPU nhất), C8, C4, C5, C6.
5. Viết D1 và D2 từ các file kết quả; sửa E2.
6. Mở PR cuối, đợi CI xanh, merge, A3.
7. Nhóm hoàn thiện mục F.

Phụ thuộc chính: B3 cần B1 và B2; D1 cần kết quả của bước 3; mọi việc ở bước 3 cần nhánh code đã đẩy lên.

## 7. Nếu một việc P1 không kịp

Không ghi kết quả thay thế. Ghi đúng tình trạng vào mục hạn chế của báo cáo cuối:

| Việc không kịp | Câu ghi trong báo cáo |
| --- | --- |
| C5 | RQ1 chỉ được trả lời cho phần tiền xử lý; phần augmentation chưa có thí nghiệm |
| C6 | Phần phân loại chi tiêu của RQ2 chưa được thực hiện |
| C7 | Kết quả từ một seed; khoảng tin cậy chỉ phản ánh việc lấy mẫu hóa đơn |
| C8 | Phân loại nguyên nhân lỗi là quy tắc tự động, chưa được kiểm tra bằng mắt |
| B4 | Demo là notebook, chưa có giao diện |

## 8. Nội dung đã có bằng chứng cho mục Hạn chế và Hướng cải tiến

Hạn chế và rủi ro (nguồn: `week4_eval.zip`, báo cáo Tuần 4):

- OCR là điểm nghẽn: khoảng 70% trường sai trên test nằm ở phía OCR hoặc nhãn; trần của ADDRESS chỉ 0.616.
- M1b hơn M1a 0.020 macro F1 trên test; kết luận này dựa trên một seed.
- Tập validation và test nhỏ (115 và 116 hóa đơn), khoảng tin cậy rộng.
- TIMESTAMP do luật ngày quyết định là chính, không phản ánh khả năng của model.
- B0 không dự đoán được SELLER (F1 0.000) và Silhouette chỉ 0.088; đây là baseline đã đơn giản hóa.
- Tập test được chấm lại một lần sau khi sửa lỗi cách đọc nhãn tổng tiền; dự đoán của model không đổi và cùng quy tắc áp dụng cho cả ba hệ thống.
- Khoảng 10% hóa đơn validation và test có bản gần trùng ở tập khác (cần file kết quả của C1 để dẫn số).
- MC-OCR thiên về chuỗi siêu thị và hóa đơn in rõ; chỉ có nhãn cho 4 trường nên ITEM_NAME, ITEM_PRICE, QUANTITY chưa được đo.
- Phân loại lỗi là quy tắc tự động.
- Môi trường: detectron2 phải build lại mỗi phiên; transformers 5.x; checkpoint M1a đã mất.

Hướng cải tiến (nguồn: kế hoạch cải tiến trong báo cáo Tuần 4):

- Fine-tune VietOCR có augmentation và chỉnh ngưỡng CRAFT.
- Chuẩn hóa nhãn tổng tiền và thêm kiểm tra chéo số tiền.
- Nhiều seed và ensemble.
- Augmentation cho bước trích xuất (giả lỗi OCR, dịch ô chữ).
- Focal loss hoặc lớp CRF cho chuỗi BIO.
- Trích ITEM_NAME, ITEM_PRICE, QUANTITY; thử trên CORD và SROIE như đã nêu ở Tuần 2.

## 9. Quyết định cần nhóm chốt

1. Trọng số M1b để private hay công khai (E5).
2. Có chạy thêm seed không (C7), vì đây là việc tốn GPU nhất.
3. Có làm phần phân loại chi tiêu không (C6), vì cần người gán nhãn 231 hóa đơn.
4. Có thử fine-tune VietOCR không (C5).
5. Có làm giao diện Gradio và video không (B4).
