import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.pipeline.demo_view import (draw_words, field_of_tag, field_rows, format_vnd, item_rows, items_summary,
                                    missing_fields, prepare_upload, timing_rows)

FULL = {
    "file_id": "r1",
    "fields": {
        "TIMESTAMP": {"text": "Ngày 15/08/2020 10:21", "date": "15/08/2020", "time": "10:21"},
        "TOTAL_COST": {"text": "585,500", "amount_vnd": 585500},
        "SELLER": {"text": "Phúc Anh Minimart"},
        "ADDRESS": {"text": "248 Cổ Bi - Gia Lâm Hà Nội"},
    },
    "items": [
        {"name": "Tã Quần Huggie", "quantity": 1, "unit_price": 318000, "amount": 318000},
        {"name": "Chupa", "quantity": 2, "unit_price": 4500, "amount": 9000},
    ],
    "items_check": {"sum": 327000, "status": "verified", "predicted_total": 327000},
}


class FormatTest(unittest.TestCase):
    def test_vnd(self):
        self.assertEqual(format_vnd(585500), "585.500 đ")
        self.assertEqual(format_vnd(0), "0 đ")
        self.assertEqual(format_vnd(None), "")

    def test_tag(self):
        self.assertEqual(field_of_tag("B-TOTAL_COST"), "TOTAL_COST")
        self.assertEqual(field_of_tag("I-SELLER"), "SELLER")
        self.assertIsNone(field_of_tag("O"))
        self.assertIsNone(field_of_tag(None))


class FieldRowsTest(unittest.TestCase):
    def test_all_fields_in_fixed_order(self):
        rows = field_rows(FULL)
        self.assertEqual([r[0] for r in rows], ["Người bán", "Địa chỉ", "Ngày giờ", "Tổng tiền"])
        self.assertEqual(rows[2][2], "15/08/2020 10:21")
        self.assertEqual(rows[3][2], "585.500 đ")
        self.assertEqual(missing_fields(FULL), [])

    def test_missing_field_has_no_row(self):
        only_total = {"file_id": "r2", "fields": {"TOTAL_COST": {"text": "418,986", "amount_vnd": 418986}}}
        self.assertEqual(field_rows(only_total), [["Tổng tiền", "418,986", "418.986 đ"]])
        self.assertEqual(missing_fields(only_total), ["Người bán", "Địa chỉ", "Ngày giờ"])

    def test_date_without_time(self):
        rec = {"fields": {"TIMESTAMP": {"text": "15/08/2020", "date": "15/08/2020"}}}
        self.assertEqual(field_rows(rec)[0][2], "15/08/2020")


class ItemsTest(unittest.TestCase):
    def test_rows(self):
        self.assertEqual(item_rows(FULL)[1], [2, "Chupa", 2, "4.500 đ", "9.000 đ", "✓"])

    def test_line_that_does_not_add_up_is_flagged(self):
        rec = {"items": [{"name": "SCA Susu", "quantity": 25, "unit_price": 25000, "amount": 5000},
                         {"name": "x", "quantity": None, "unit_price": None, "amount": 1000}]}
        self.assertEqual([r[-1] for r in item_rows(rec)], ["⚠", ""])

    def test_no_items(self):
        self.assertEqual(item_rows({"fields": {}}), [])
        self.assertIn("Không đọc được", items_summary({"fields": {}}))

    def test_verified_and_unverified(self):
        self.assertTrue(items_summary(FULL).startswith("✅"))
        bad = dict(FULL, items_check={"sum": 565500, "status": "unverified", "predicted_total": 585500})
        text = items_summary(bad)
        self.assertTrue(text.startswith("⚠️"))
        self.assertIn("565.500 đ", text)
        self.assertIn("585.500 đ", text)


class DrawTest(unittest.TestCase):
    def test_boxes_coloured_by_field_and_clipped(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        words = [{"text": "a", "bbox": [10, 10, 50, 30], "tag": "B-TOTAL_COST"},
                 {"text": "b", "bbox": [60, 10, 90, 30], "tag": "O"},
                 {"text": "c", "bbox": [150, 80, 400, 300], "tag": "I-SELLER"}]   # past the edge
        out = draw_words(img, words)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue((img == 255).all())                       # input untouched
        self.assertEqual(tuple(out[10, 30]), (40, 40, 220))       # red top edge of the total
        self.assertEqual(tuple(out[10, 75]), (170, 170, 170))     # grey for O
        self.assertEqual(tuple(out[99, 170]), (40, 120, 230))     # clipped seller box still drawn

    def test_no_words(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        self.assertTrue((draw_words(img, []) == img).all())


class UploadTest(unittest.TestCase):
    def test_exif_rotation_and_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "phone.jpg")
            im = Image.new("RGB", (40, 20), "white")
            exif = im.getexif()
            exif[0x0112] = 6                                       # orientation: rotate 90 degrees
            im.save(src, exif=exif)
            dst = prepare_upload(src, os.path.join(tmp, "up.jpg"))
            with Image.open(dst) as out:
                self.assertEqual(out.size, (20, 40))
                self.assertEqual(out.mode, "RGB")

    def test_upright_rgb_jpeg_is_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "receipt.jpg")
            Image.new("RGB", (30, 50), "white").save(src)
            dst = os.path.join(tmp, "up.jpg")
            self.assertEqual(prepare_upload(src, dst), src)
            self.assertFalse(os.path.exists(dst))

    def test_png_with_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shot.png")
            Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(src)
            with Image.open(prepare_upload(src, os.path.join(tmp, "up.jpg"))) as out:
                self.assertEqual(out.mode, "RGB")


class TimingTest(unittest.TestCase):
    def test_known_stages_in_order(self):
        rows = timing_rows({"total": 2.9, "ocr": 1.27, "preprocess": 0.08, "features": 0.001})
        self.assertEqual(rows, [["Tiền xử lý ảnh", "0.08 s"], ["VietOCR đọc chữ", "1.27 s"], ["Tổng", "2.90 s"]])


if __name__ == "__main__":
    unittest.main()
