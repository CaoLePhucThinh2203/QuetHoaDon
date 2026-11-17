import unittest

from src.extraction.line_items import check_items, extract_items, group_lines, parse_amounts, plain


def lay(lines):
    """Words and boxes of text lines stacked every 40 units (x grows with the word index)."""
    words, boxes = [], []
    for row, text in enumerate(lines):
        for col, w in enumerate(text.split()):
            words.append(w)
            boxes.append([10 + 60 * col, 40 * row, 60 + 60 * col, 40 * row + 20])
    return words, boxes


class ParseAmountsTest(unittest.TestCase):
    def values(self, text):
        return [a[0] for a in parse_amounts(text)]

    def test_separator_styles(self):
        self.assertEqual(self.values("36 .500 36 .500"), [36500, 36500])
        self.assertEqual(self.values("24 000"), [24000])
        self.assertEqual(self.values("GOI 4,000 8, 000"), [4000, 8000])
        self.assertEqual(self.values("110,000 110, 000"), [110000, 110000])
        self.assertEqual(self.values("645 .000đ"), [645000])

    def test_decimal_tail_and_barcode(self):
        self.assertEqual(self.values("5, 800 .00"), [5800])
        self.assertEqual(self.values("8934804025650 36 .500"), [36500])

    def test_plain_numbers_are_not_amounts(self):
        self.assertEqual(self.values("0869322496 02438765210"), [])
        self.assertEqual(self.values("Số 590 Trần Phú"), [])

    def test_discount_sign(self):
        a = parse_amounts("KM -3 .300")
        self.assertEqual((a[0][0], a[0][3]), (3300, True))


class GroupLinesTest(unittest.TestCase):
    def test_words_in_reading_order(self):
        words, boxes = lay(["a b", "c d e"])
        self.assertEqual(group_lines(words[::-1], boxes[::-1]), ["a b", "c d e"])

    def test_plain(self):
        self.assertEqual(plain("TỔNG TIỀN PHẢI T.TOÁN"), "tong tien phai t toan")


class ExtractItemsTest(unittest.TestCase):
    def test_two_line_layout_with_barcode(self):
        words, boxes = lay(["VinCommerce", "Mặt hàng đơn giá SL T.Tiền", "MIWON Lá Kim cuốn cơm 10g",
                            "8935304200233 22 .500 22 .500", "sữa TT Vinami 1k không đường túi 220m",
                            "8934673501354 7 .200 2 14 .400", "TỔNG TIỀN PHẢI T. TOÁN 36 .900"])
        res = extract_items(words, boxes)
        self.assertTrue(res["header_found"])
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"]) for i in res["items"]],
                         [("MIWON Lá Kim cuốn cơm 10g", 1, 22500, 22500),
                          ("sữa TT Vinami 1k không đường túi 220m", 2, 7200, 14400)])
        chk = check_items(res["items"], res["summary_amounts"])
        self.assertEqual((chk["sum"], chk["status"]), (36900, "verified"))

    def test_one_line_item_and_summary_line_is_not_an_item(self):
        words, boxes = lay(["TT Tên món SL Đơn giá Thành tiền", "1 Cafe sữa M 24 000 24 000", "Thành tiền: 24 000",
                            "Tổng cộng: 24 000"])
        items = extract_items(words, boxes)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0]["name"], items[0]["amount"]), ("Cafe sữa M", 24000))

    def test_price_line_under_the_name_and_registered_price_skipped(self):
        words, boxes = lay(["Tên hàng SL ĐVT Đơn giá Thành tiền", "Sữa Mộc Châu đường 180ml NT",
                            "Giá đăng ký: 7,000", "Giá bán: 12 HOP 6,500 78,000", "Tổng cộng: 78,000"])
        items = extract_items(words, boxes)["items"]
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"]) for i in items],
                         [("Sữa Mộc Châu đường 180ml", 12, 6500, 78000)])

    def test_discount_line_is_subtracted(self):
        words, boxes = lay(["Tên hàng SL Thành tiền", "Trà túi lọc Lipton", "38 .000 38 .000", "KM -3 .300", "Tổng cộng 34 .700"])
        res = extract_items(words, boxes)
        chk = check_items(res["items"], res["summary_amounts"])
        self.assertEqual((chk["sum"], chk["status"]), (34700, "verified"))

    def test_nothing_to_read(self):
        words, boxes = lay(["Cửa hàng A", "Địa chỉ B"])
        res = extract_items(words, boxes)
        self.assertEqual(res["items"], [])
        self.assertEqual(check_items([], [None])["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
