# Báo cáo Tuần 3 — Đánh giá hiện trạng

*Cập nhật dựa trên `AI project.docx` (kế hoạch + trình bày Tuần 1-3) và repo tại commit `1f0d8dd` (04/11/2026).*

## 1. Kết luận

**Tuần 3 chưa hoàn thành.** Đầu ra Tuần 3 là *"một dataset pipeline hoàn chỉnh và hướng mô hình có lập luận"*.
Phần dữ liệu và phần lập luận chọn mô hình đã có; phần **feature engineering** và **baseline metrics** chưa có;
pipeline ảnh mới chạy được ở mức mẫu.

| # | Hạng mục Tuần 3 | Trạng thái | Bằng chứng | Còn thiếu |
| --- | --- | --- | --- | --- |
| 1 | Data preprocessing pipeline | **Một phần** | Parser + `dataset_preparer` (split theo hóa đơn 922/115/116, leakage check, manifest) có test; Stage 0 (resize, deskew, CLAHE) chạy trong notebook | Chưa xử lý ảnh trùng / gần trùng (plan tự nêu rủi ro); notebook xử lý 1044 ảnh nhưng có 1153 hóa đơn hợp lệ, chưa đối chiếu; notebook không dùng manifest/split; chưa có bước lọc ảnh lỗi/mờ như plan §1.1 |
| 2 | Feature engineering | **Chưa** | Plan liệt kê 6 nhóm đặc trưng; code chỉ tạo embedding LayoutXLM + bbox chuẩn hóa | Chưa có code cho vị trí dọc, căn hàng, pattern số/tiền/ngày, quan hệ giữa vùng chữ |
| 3 | Model selection justification | **Gần xong** | Bảng chọn CRAFT / VietOCR / LayoutXLM + lý do trong plan và `docs/architecture.md` | Mâu thuẫn chưa chốt: Tuần 2 nói LayoutLM *trích xuất* trường, Tuần 3 nói LayoutXLM chỉ *trích đặc trưng* + clustering; chưa so sánh với phương án thay thế |
| 4 | Baseline model metrics | **Chưa** | Chỉ có thống kê dữ liệu, không có số đo mô hình nào (CER/WER, detection, F1, Silhouette, EM, thời gian) | Toàn bộ; lần chạy full gần nhất ghi nhận 5/1044 ảnh thành công do lỗi `inhomogeneous shape` |
| 5 | Planned modelling tasks cho Tuần 4 | **Một phần** | Danh sách 9 việc trong handover cũ, không có người phụ trách / hạn / tiêu chí | Đã bổ sung trong `reports/week4_plan.md` |

## 2. Sai sót / thiếu sót trong repo

Mục "Xử lý" cho biết đã sửa trong lần sắp xếp lại này hay còn là việc cần làm.

| # | Vấn đề | Bằng chứng | Xử lý |
| --- | --- | --- | --- |
| 1 | README trỏ tới `notebooks/kaggle/01_mc_ocr_eda.ipynb` nhưng commit `1f0d8dd` đã xóa file này; bản cũ cũng chưa từng chạy (0 output) | `git show --stat 1f0d8dd` | Khôi phục thành `notebooks/01_mc_ocr_eda.ipynb`. **Cần chạy và lưu output** |
| 2 | Toàn bộ logic pipeline nằm trong notebook 501 KB (444 KB là output pip/ảnh); `src/detection`, `src/recognition` chỉ là interface không ai dùng | grep không thấy nơi nào import | Chuyển vào `src/` (có test), notebook còn 8 KB |
| 3 | `configs/*.yaml` không được code nào đọc; notebook hard-code tham số | grep `yaml` trong `src/` và notebook | Thêm `src/common/config.py`; notebook đọc từ `configs/` |
| 4 | Định dạng bbox không thống nhất: `polygon_to_bbox`, detector → `xywh`; docs, IoU, notebook → `xyxy` | `bbox_utils.py`, `detector.py` | Quy ước `xyxy` nội bộ, thêm `xywh_to_xyxy`… (MC-OCR gốc là `xywh`) |
| 5 | `requirements.txt` thiếu `craft-text-detector`, `detectron2`, `kagglehub`, `matplotlib`; thừa `datasets`, `pydantic`, `hdbscan`, `seqeval` (chưa dùng) | so với import thực tế | Viết lại; thêm `requirements-dev.txt` cho test/CI |
| 6 | `.gitignore` không chặn `pipeline_results/` (hàng nghìn ảnh + JSON) | notebook ghi vào `./pipeline_results` | Đã thêm |
| 7 | Không có CI | không có `.github/workflows` | Thêm workflow chạy unit test |
| 8 | Cùng một pipeline được mô tả 4-5 lần (README, architecture, preprocessing, handover, week3 report) và dễ lệch nhau | so sánh nội dung | Gộp còn `docs/architecture.md`, `docs/dataset.md` |
| 9 | Mâu thuẫn chính sách: README/config nói "unsupervised, nhãn chỉ để đánh giá", nhưng repo lại chia train/val/test (vô nghĩa nếu không huấn luyện); plan Tuần 2 lại dùng LayoutLM để trích xuất | `dataset_preparer.py`, plan §Baseline | **Cần nhóm quyết** — `week4_plan.md` mục 0 |
| 10 | Mục tiêu `ITEM_NAME`/`ITEM_PRICE`/`QUANTITY` **không có nhãn** trong MC-OCR (chỉ có SELLER, ADDRESS, TIMESTAMP, TOTAL_COST) nên không thể tính F1 cho các trường này | `annotation_parser.py`, EDA | Ghi rõ trong docs; đánh giá định tính |
| 11 | Thống kê trong plan: `label_distribution` cộng 3033 nhưng `annotated_regions` = 6585 → thiếu TIMESTAMP/TOTAL_COST | plan §Target Information | Chạy lại `scripts/prepare_dataset.py`, ghi đủ 4 nhãn |
| 12 | Manifest lưu `image_path` tuyệt đối (`/kaggle/input/...`) | `dataset_preparer.py` | **Còn lại** (việc Tuần 4) |
| 13 | `TextRecognitionResult.confidence` bắt buộc là `float`, docs nói có thể `null` | `recognizer.py` | Đổi thành `Optional[float]` |
| 14 | Lỗi `inhomogeneous shape` làm 1039/1044 ảnh fail; chạy 10 phút mới biết | log chạy | Ép CRAFT trả box 4 điểm; runner tự dừng sớm; **chưa chạy lại trên Colab** |
| 15 | CRAFT tách riêng box cho dấu phẩy/chấm trong `27,000`, VietOCR đọc ra `9`, `D`, `2`, `5` | output notebook (box 16, 17, 20, 23, 25) | `drop_nested`; **chưa kiểm chứng trên Colab** |
| 16 | Leakage chỉ kiểm theo `img_id`, không phát hiện ảnh chụp lại cùng hóa đơn | `check_data_leakage` | **Còn lại** (Tuần 4, WS-A) |

## 3. Đánh giá model hiện tại

Model hiện tại = pipeline pretrained **CRAFT → VietOCR → LayoutXLM (trích đặc trưng)**, chưa huấn luyện thêm.

**Đã đạt được** (bằng chứng: 1 hóa đơn mẫu trong notebook và log nạp model):

- Ba model nạp được trên GPU (LayoutXLM 744/744 trọng số; cảnh báo `UNEXPECTED` chỉ là bộ đếm batch-norm, vô hại).
- Trên hóa đơn mẫu: 31 vùng được phát hiện (5 trong số đó là box dấu phẩy/chấm rác, đã có bản sửa), box bao sát từng dòng kể cả logo; thứ tự đọc đúng; chữ in rõ được đọc với độ tin cậy ≈ 0.93.
- Pipeline có resume, ghi file nguyên tử, tự dừng sớm, đo thời gian từng stage (phục vụ RQ3).

**Chưa đạt:**

- Chưa có **đầu ra có cấu trúc**: hệ thống chưa trả về seller / address / timestamp / total.
- Chưa có **bất kỳ số đo nào** (xem hạng mục 4 ở trên); chưa chứng minh chạy hết 1044 ảnh.
- Chưa trả lời được RQ1 (ảnh hưởng của tiền xử lý — chưa có đối chứng bật/tắt), RQ2 (kiến trúc nào tối ưu), RQ3 (độ chính xác ↔ độ trễ).

**Cần cải tiến** (từ quan sát trên mẫu và thiết kế):

1. **Trích xuất trường**: cần tầng quyết định ý nghĩa (luật hoặc phân loại token). Embedding + clustering không tự gán nhãn `TOTAL_COST`.
2. **OCR**: `(+VAT)` bị đọc thành `((-VAT)`; box nhiễu cho confidence thấp (0.17); nên lọc theo `ocr_confidence` và hậu xử lý số tiền.
3. **Detection**: cột số lượng `1` không có box; `27,000 Tien thu` bị gộp thành một vùng → nhãn trường sẽ lẫn nhau.
4. **LayoutXLM**: giới hạn 512 token, hóa đơn dài bị cắt (đã có cờ `possibly_truncated`, chưa có cửa sổ trượt).
5. **Tiền xử lý**: CLAHE + bilateral có thể làm mờ nét in nhiệt — cần thí nghiệm bật/tắt (RQ1).
6. **Dữ liệu**: ảnh gần trùng giữa các split sẽ làm số đo ảo.

## 4. Việc tiếp theo

Xem `reports/week4_plan.md`.
