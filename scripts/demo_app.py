"""Web demo: upload a receipt photo, get the four fields and the item lines (Gradio, run on a Kaggle / Colab GPU).

The model is the frozen final configuration, built exactly as in ``scripts/extract_receipt.py`` and notebook 07:
preprocessing -> CRAFT (nested-box filter 0.5) -> fine-tuned VietOCR -> M1b v3 -> date-time rules and SELLER / ADDRESS
clean-up -> rule-based item reader. The page shows

* the preprocessed receipt with every OCR word box, coloured by the field the model gave it;
* a table with only the fields that were read, and the item lines with the check against the printed total;
* the compact JSON (``compact_result``) and the seconds spent in each stage.

Example (Kaggle, see ``notebooks/08_web_demo_kaggle.ipynb``):
    python scripts/demo_app.py --checkpoint m1b_v3_checkpoint/best --ocr-weights vietocr_finetuned.pth \
        --examples ex1.jpg ex2.jpg --share

``--share`` prints a public ``https://....gradio.live`` link that works while the process runs (up to 72 hours).
"""
import argparse
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.demo_view import (field_rows, draw_words, item_rows, items_summary, legend_markdown,  # noqa: E402
                                    missing_fields, prepare_upload, timing_rows)
from src.pipeline.extractor import ReceiptExtractor, compact_result  # noqa: E402

INTRO = """# QuetHoaDon · Quét hóa đơn tiếng Việt
Tải ảnh hóa đơn lên (hoặc chọn một ảnh mẫu bên dưới) rồi bấm **Quét hóa đơn**. Hệ thống đọc **người bán, địa chỉ, ngày giờ,
tổng tiền** và **từng mặt hàng**; trường nào không đọc được thì không hiện.

Pipeline: tiền xử lý → CRAFT → VietOCR tinh chỉnh → LayoutXLM (M1b v3) → luật ngày giờ, làm sạch → đọc mặt hàng.
Trên tập test MC-OCR, đúng tuyệt đối: người bán 72,0 %, địa chỉ 27,4 %, ngày 84,0 %, tổng tiền 72,4 %; cả bốn trường
18,4 %. Hãy kiểm tra lại kết quả, đặc biệt địa chỉ. Ảnh mẫu lấy từ tập test MC-OCR 2021.
"""


FIELD_HEADERS = ["Trường", "Giá trị đọc được", "Chuẩn hóa"]
ITEM_HEADERS = ["#", "Tên hàng", "SL", "Đơn giá", "Thành tiền", "SL×giá"]
TIMING_HEADERS = ["Khâu", "Thời gian"]


def safe_stem(path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem).strip("_.")
    return stem or "receipt"


def make_scan(extractor: ReceiptExtractor):
    import cv2
    import pandas as pd

    def scan(image_path):
        empty_fields = pd.DataFrame(columns=FIELD_HEADERS)
        empty_items = pd.DataFrame(columns=ITEM_HEADERS)
        if not image_path:
            return None, "Chưa có ảnh: hãy tải ảnh hóa đơn lên.", empty_fields, "", empty_items, {}, None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                upright = prepare_upload(image_path, os.path.join(tmp, safe_stem(image_path) + "_in.jpg"))
                meta, image = extractor.ocr_pipeline.infer(upright)
                meta["file_id"] = safe_stem(image_path)
                pre = os.path.join(tmp, meta["file_id"] + ".jpg")
                cv2.imwrite(pre, image, [cv2.IMWRITE_JPEG_QUALITY, 95])   # same encoding as ReceiptExtractor.extract
                result = extractor.extract_from_ocr(meta, pre)
        except Exception:                                                  # show the error instead of a blank page
            return None, "Lỗi khi xử lý ảnh:\n```\n" + traceback.format_exc()[-1500:] + "\n```", empty_fields, "", \
                empty_items, {}, None

        compact = compact_result(result)
        annotated = cv2.cvtColor(draw_words(image, result["words"]), cv2.COLOR_BGR2RGB)
        rows = field_rows(compact)
        missing = missing_fields(compact)
        status = f"Đọc được {len(rows)} / 4 trường, {result['n_words']} từ OCR."
        if missing:
            status += " Không đọc được: " + ", ".join(missing) + "."
        fields_df = pd.DataFrame(rows, columns=list(empty_fields.columns)) if rows else empty_fields
        items = item_rows(compact)
        items_df = pd.DataFrame(items, columns=list(empty_items.columns)) if items else empty_items
        timing_df = pd.DataFrame(timing_rows(result["timing_sec"]), columns=TIMING_HEADERS)
        return annotated, status, fields_df, items_summary(compact), items_df, compact, timing_df

    return scan


def build_ui(scan, examples):
    import gradio as gr

    with gr.Blocks(title="QuetHoaDon · Quét hóa đơn") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                # image_mode=None: gradio hands over the uploaded file itself instead of a re-encoded copy
                inp = gr.Image(type="filepath", image_mode=None, label="Ảnh hóa đơn", sources=["upload", "clipboard"],
                               height=480)
                btn = gr.Button("Quét hóa đơn", variant="primary")
                if examples:
                    gr.Examples(examples=[[e] for e in examples], inputs=inp, label="Ảnh mẫu (tập test MC-OCR)")
            with gr.Column(scale=1):
                annotated = gr.Image(label="Ảnh sau tiền xử lý và các từ đã đọc", height=480, interactive=False)
                gr.Markdown(legend_markdown())
        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Thông tin hóa đơn")
                fields = gr.Dataframe(headers=FIELD_HEADERS, interactive=False, wrap=True,
                                      column_widths=["22%", "50%", "28%"])
            with gr.Column(scale=1):
                gr.Markdown("### Mặt hàng")
                summary = gr.Markdown()
                items = gr.Dataframe(headers=ITEM_HEADERS, interactive=False, wrap=True,
                                     column_widths=["6%", "40%", "8%", "17%", "17%", "12%"])
                gr.Markdown("Cột SL×giá: ✓ khi số lượng × đơn giá = thành tiền, ⚠ khi không khớp (có số đọc sai).")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("JSON đầu ra", open=False):
                    js = gr.JSON()
            with gr.Column(scale=1):
                with gr.Accordion("Thời gian từng khâu", open=False):
                    timing = gr.Dataframe(headers=TIMING_HEADERS, interactive=False)
        outputs = [annotated, status, fields, summary, items, js, timing]
        btn.click(scan, inputs=inp, outputs=outputs)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="M1b v3 'best' folder")
    ap.add_argument("--ocr-weights", required=True, help="fine-tuned VietOCR weights (vietocr_finetuned.pth)")
    ap.add_argument("--config-dir", default=str(Path(__file__).resolve().parent.parent / "configs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nested-threshold", type=float, default=0.5, help="final configuration: 0.5")
    ap.add_argument("--examples", nargs="*", default=[], help="example receipt images shown under the upload box")
    ap.add_argument("--share", action="store_true", help="public gradio.live link (Kaggle / Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("[demo] loading the final configuration ...", flush=True)
    extractor = ReceiptExtractor.from_checkpoint(args.checkpoint, args.config_dir, args.device, with_ocr=True,
                                                 postprocess=True, ocr_weights=args.ocr_weights, items=True,
                                                 nested_threshold=args.nested_threshold)
    examples = [e for e in args.examples if os.path.exists(e)]
    demo = build_ui(make_scan(extractor), examples)
    demo.queue(max_size=16).launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
