# Tình hình dự án — cập nhật 08/11/2026

> **Ghi chú 10/11/2026:** nhóm đã **bỏ notebook DesignThinking** khỏi kế hoạch. Các số ở mục 3 và baseline "luật + LogisticRegression"
> chỉ giữ để tham khảo, **không dùng làm baseline**. Baseline chính thức B0 là pipeline Tuần 3 (đặc trưng LayoutXLM đóng băng,
> phân cụm, gán cụm vào trường kèm luật); model cải tiến là M1b (LayoutXLM fine-tune). Mục 4 và 6 bên dưới đã cũ: kế hoạch
> hiện hành và bảng khớp với docx nằm ở [`project_plan.md`](project_plan.md).

*Tổng hợp từ repo (commit `6449de5`), `AI Project Progress.docx` (bản trình bày Tuần 1-3) và notebook
`QuetHoaDon_DesignThinking.ipynb` (chạy trên Colab GPU, **chưa có trong repo**). Bổ sung cho
[`week3_progress_report.md`](week3_progress_report.md) và [`week4_plan.md`](week4_plan.md), không thay thế.*

## 1. Tóm tắt

- Tuần 1-2 xong. Tuần 3 đã viết đủ trong docx nhưng code còn thiếu feature engineering và chưa có số đo trong repo.
  Tuần 4 (huấn luyện + đánh giá) **chưa bắt đầu** trong repo.
- **Có tin mới:** notebook DesignThinking đã chạy Stage 0-2 (CRAFT + VietOCR) trên **1044/1044 ảnh, 0 lỗi**, và đã có
  **số liệu baseline đầu tiên** (luật + LogisticRegression). Đây là ứng viên cho baseline **B0** của Tuần 4.
- **Hướng đi đề xuất:** giữ LayoutXLM, đi theo 3 nấc B0 → M1a (LayoutXLM đóng băng + head) → M1b (fine-tune LayoutXLM).
  Chi tiết ở mục 4. Nhóm cần chốt.

## 2. Tiến độ theo tuần

| Tuần | Đầu ra | Trạng thái |
| --- | --- | --- |
| 1 | SWOT, communication plan, overview, WBS, cấu trúc repo | Xong |
| 2 | Problem statement, 3 RQ, khảo sát dataset, data quality review, baseline solution | Xong trong docx; problem statement + RQ **nay đã đưa vào README** |
| 3 | Preprocessing pipeline, feature engineering, chọn model, baseline metrics, plan Tuần 4 | Docx đã viết đủ. Repo: preprocessing một phần, feature engineering chưa có code, metrics mới có trong notebook ngoài repo |
| 4 | Kiến trúc model, training, metrics, so sánh baseline vs improved, phân tích lỗi | Chưa bắt đầu trong repo (kế hoạch: `week4_plan.md`) |
| 5 | Demo end-to-end, tổng kết, hạn chế, hướng cải tiến, review GitHub | Chưa |

## 3. Kết quả từ notebook DesignThinking (ngoài repo)

Notebook gồm Stage 0-3 giống pipeline trong `src/` (notebook nháp `QuetHoaDon.ipynb` đã xóa) (LayoutXLM **tắt**), cộng thêm Stage 4-7:

| Stage | Nội dung |
| --- | --- |
| 4A | Nạp annotation MC-OCR: 1937 dòng từ 3 file CSV, parse được nhãn của 1153 ảnh |
| 4B | Gộp ô OCR thành hàng, gán nhãn hàng bằng so khớp văn bản (difflib) |
| 4C | Chia train/val/test theo hóa đơn + LogisticRegression one-vs-rest trên 22 đặc trưng hàng |
| 4D / 4E | Đánh giá đầu-cuối, phân tích lỗi tổng tiền |
| 5 | Phân loại chi tiêu food / move / study / shop / other (TF-IDF char n-gram + LR) |
| 6 | Xuất `receipt_model.json` (197 KB) + `receipt_lexicon.json` |
| 7 | Đánh giá trên OCR của ML Kit: chưa có mẫu |

### Số liệu

**Chạy OCR:** 1044 ảnh, 1044 thành công, 2256 s (≈ 2.2 s/ảnh trên GPU Colab). Lỗi `inhomogeneous shape` ghi trong
`week3_progress_report.md` không xuất hiện trong lần chạy này.

**Dữ liệu cấp hàng:** 1043 hóa đơn, 23 546 hàng, 922 hóa đơn có tổng tiền chuẩn. Phân bố nhãn hàng:
OTHER 20 186 · ADDRESS 1355 · SELLER 973 · TIMESTAMP 690 · **TOTAL_COST 342**.

**Phân loại hàng (validation, 105 hóa đơn):**

| Trường | Precision | Recall | F1 |
| --- | --- | --- | --- |
| SELLER | 0.819 | 0.584 | 0.682 |
| ADDRESS | 0.698 | 0.536 | 0.607 |
| TIMESTAMP | 0.949 | 1.000 | 0.974 |
| TOTAL_COST | 0.727 | 0.485 | 0.582 |

**Đánh giá đầu-cuối:**

| Tập | Chế độ | Tổng tiền đúng | Tổng tiền trong top-3 | Ngày đúng | Seller (độ giống ≥ 0.8) | n |
| --- | --- | --- | --- | --- | --- | --- |
| val | luật | 0.258 | 0.278 | 0.798 | 0.485 | 97 |
| val | luật + LR | 0.309 | 0.340 | 0.798 | 0.717 | 97 |
| test | luật | 0.233 | 0.267 | 0.708 | 0.474 | 90 |
| test | luật + LR | **0.322** | 0.344 | 0.708 | 0.629 | 90 |

**Lỗi tổng tiền điển hình (Stage 4E):** lấy nhầm mã nhân viên (NVBH) hoặc pass wifi thành tổng tiền; lấy "tiền khách
trả" thay cho tổng; OCR tách `107,000` thành `107 7,000`; box gộp nhiều số trên một dòng.

**Phân loại chi tiêu:** độ chính xác 0.846 trên test, nhưng học và chấm **trên nhãn yếu sinh từ từ điển** → chỉ chứng
minh code chạy, **không phải độ chính xác thật**. Cần khoảng 200 nhãn gán tay (`category_labels_fixed.csv`).

### Giới hạn khi dùng các số này

1. **Split khác repo:** notebook tự chia 834 / 105 / 104 (GroupShuffleSplit) trên 1043 hóa đơn; repo chia
   922 / 115 / 116 trên 1153 hóa đơn. Phải chạy lại B0 trên split của repo trước khi so với model khác.
2. **Trần nhãn TOTAL_COST thấp:** 922 hóa đơn có tổng tiền nhưng so khớp văn bản chỉ gán được 342 hàng → recall bị chặn
   trên ngay từ bước gán nhãn.
3. **Không phải deep learning:** LR trên đặc trưng thủ công → hợp làm baseline B0, không đủ cho mục 3 của đề (5 điểm).
4. Khoảng cách 1044 / 1153 nhiều khả năng do `locate_images()` chỉ lấy ảnh trong thư mục `data0.7`; chưa kiểm chứng.

## 4. Hướng đi đề xuất (giữ LayoutXLM)

Docx Tuần 3 dùng LayoutXLM để **trích đặc trưng** rồi phân cụm (Silhouette). Hướng này không có bước
train/validate/improve và Silhouette không đo được việc trích đúng tổng tiền, nên rủi ro với mục 3 của đề. Đề xuất giữ
vai trò "trích đặc trưng" nhưng thay phân cụm bằng huấn luyện có giám sát, theo 3 nấc:

| Mã | Model | Huấn luyện | Vai trò |
| --- | --- | --- | --- |
| **B0** | Luật + LogisticRegression (từ notebook DesignThinking) | LR trên train | Baseline, đã có số |
| **M1a** | LayoutXLM **đóng băng** + head MLP phân loại token | chỉ head | Thay phân cụm; dùng lại code Stage 3; nhẹ |
| **M1b** | **Fine-tune** `LayoutXLMForTokenClassification` (BIO, 9 nhãn) + hậu xử lý TOTAL_COST | toàn bộ | Model cải tiến chính (= M1 trong `week4_plan.md`) |

Thí nghiệm cho từng RQ:

- **RQ1:** cắt các vùng có nhãn trong MC-OCR (có sẵn text chuẩn), đo **CER của VietOCR khi bật / tắt tiền xử lý**.
  Không phụ thuộc detection. Thêm augmentation khi fine-tune M1b.
- **RQ2:** so B0 / M1a / M1b bằng F1 từng trường + tổng tiền đúng, trên **cùng một split**.
- **RQ3:** thời gian/ảnh theo stage, kích thước model, GPU vs CPU. B0 đủ nhẹ để chạy trên máy; LayoutXLM chạy phía server
  hoặc làm model "thầy".

Lưu ý kỹ thuật: nhãn token lấy bằng ghép box CRAFT với polygon MC-OCR theo IoU (thay cho so khớp văn bản) và báo cáo
trần recall; LayoutXLM cần detectron2 → chỉ chạy trên Colab; thống kê tỉ lệ hóa đơn vượt 512 token trước khi quyết
định có cần cửa sổ trượt.

**Về tên gọi:** notebook DesignThinking gọi hướng "ML Kit + luật + LR chạy trên điện thoại" là *phương án B*, còn
`week4_plan.md` gọi *fine-tune LayoutXLM* là *phương án B*. Đề xuất từ nay chỉ dùng các mã B0 / M1a / M1b.

## 5. Mâu thuẫn còn tồn đọng

| # | Vấn đề | Cần làm |
| --- | --- | --- |
| 1 | `configs/config.yaml` và `docs/dataset.md` vẫn ghi "nhãn chỉ để đánh giá", trong khi B0 và M1 đều train trên nhãn | Sau khi nhóm chốt: đổi `data_policy`, sửa docs |
| 2 | Docx: `label_distribution` chỉ có ADDRESS + SELLER (3033 ≠ 6585) | Chạy lại `scripts/prepare_dataset.py` trên Colab, sửa docx |
| 3 | Docx Tuần 2 nói LayoutLM *trích xuất*, Tuần 3 nói LayoutXLM *trích đặc trưng* + phân cụm | Thống nhất theo mục 4 |
| 4 | Docx mục 4.2 bỏ TIMESTAMP, có ITEM_NAME / ITEM_PRICE / QUANTITY (không có nhãn) | Thống nhất 4 trường có nhãn + item lines đánh giá định tính |
| 5 | Docs cũ theo hướng unsupervised: `docs/dataset_documentation.md`, `docs/multi_dataset_architecture.md`, `docs/preprocessing.md`, `reports/project_leader_handover.md` | **Đã xóa** khỏi repo (còn trong lịch sử Git) |
| 6 | Notebook DesignThinking chưa có trong repo nhưng là nguồn của số B0 | Đưa vào `notebooks/` hoặc port Stage 4 vào `src/extraction/` |
| 7 | Chưa kiểm tra ảnh gần trùng giữa các split; manifest lưu đường dẫn tuyệt đối | Như WS-A trong `week4_plan.md` |

## 6. Việc tiếp theo (thứ tự ưu tiên)

1. **Họp chốt** hướng B0 → M1a → M1b và phân công WS-A..D (`week4_plan.md` mục 4 chưa có tên người).
2. **WS-A:** chạy lại `prepare_dataset.py`, giải thích 1044 / 1153, kiểm tra ảnh gần trùng, viết `build_token_dataset.py` (IoU → BIO).
3. **WS-B:** port Stage 4 của notebook vào `src/extraction/` + test, chạy lại B0 trên split của repo, viết `scripts/evaluate_pipeline.py`.
4. **WS-C:** `src/training/` + `notebooks/03_train_layoutxlm.ipynb`: M1a rồi M1b (cấu hình khởi điểm ở `week4_plan.md` mục 3).
5. **WS-D:** thí nghiệm RQ1 (CER bật / tắt tiền xử lý), RQ3 (latency); viết `docs/training.md`, báo cáo đánh giá và phân tích lỗi.

WS-A và WS-B làm song song được ngay; WS-C cần token dataset của WS-A.

**Quy ước chạy:** test, pipeline và huấn luyện chạy trên **Colab**; máy local chỉ dùng để sửa code và quản lý git.
