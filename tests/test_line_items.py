import unittest
from unittest import mock

from src.extraction.line_items import (check_items, extract_items, group_lines, merge_split_numbers, parse_amounts,
                                       plain, text_slope)


def lay(lines):
    """Words and boxes of text lines stacked every 40 units (x grows with the word index)."""
    words, boxes = [], []
    for row, text in enumerate(lines):
        for col, w in enumerate(text.split()):
            words.append(w)
            boxes.append([10 + 60 * col, 40 * row, 60 + 60 * col, 40 * row + 20])
    return words, boxes


def tilt(boxes, slope):
    """The same boxes on a receipt photographed at a slant: y moves by ``slope`` per unit of x."""
    return [[x0, y0 + slope * x0, x1, y1 + slope * x0] for x0, y0, x1, y1 in boxes]


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

    def test_straight_receipt_has_no_slope(self):
        _, boxes = lay(["Tên hàng SL Đơn giá Thành tiền", "Sữa tươi Milo 180 ml", "8 6,500 52,000",
                        "Cộng tiền hàng 52,000"])
        self.assertEqual(text_slope(boxes), 0.0)

    def test_slanted_receipt_keeps_its_lines(self):
        # Train receipts such as ...014uqaoc / ...013jyzhp: the text rises to the right, so grouping by plain y puts
        # the right end of a line with the line above.
        lines = ["Tên hàng SL Đơn giá Thành tiền", "Sữa tươi Milo 180 ml", "8 6,500 52,000", "Cộng tiền hàng 52,000"]
        words, boxes = lay(lines)
        slanted = tilt(boxes, -0.1)
        with mock.patch("src.extraction.line_items.text_slope", return_value=0.0):
            self.assertNotEqual(group_lines(words, slanted), lines)          # the old grouping by plain y
        self.assertAlmostEqual(text_slope(slanted), -0.1, delta=0.025)   # a flat profile peak with only 18 words
        self.assertEqual(group_lines(words, slanted), lines)


class MergeSplitNumbersTest(unittest.TestCase):
    def merged(self, *tokens):
        return [w for w, _ in merge_split_numbers(tokens)]

    def test_repeated_digit_before_a_separator(self):
        # train ...013msokr style: "399" + "9,000" read from two overlapping crops of "399,000"
        self.assertEqual(self.merged(("399", [528, 616, 559, 631]), ("9,000", [555, 617, 594, 633])), ["399,000"])

    def test_repeated_separator(self):
        # train ...014khrvg: "6," + ",250" of "6,250"
        self.assertEqual(self.merged(("6,", [421, 572, 446, 595]), (",250", [439, 566, 512, 598])), ["6,250"])

    def test_space_thousands_and_far_apart_boxes_stay(self):
        # "10 000" is one amount written with a space; the "0" is not a character both crops saw
        self.assertEqual(self.merged(("10", [100, 10, 130, 30]), ("000", [128, 10, 170, 30])), ["10", "000"])
        self.assertEqual(self.merged(("25", [100, 10, 130, 30]), ("5.000", [200, 10, 260, 30])), ["25", "5.000"])
        self.assertEqual(self.merged(("Giá", [10, 10, 60, 30]), ("5,000", [55, 10, 120, 30])), ["Giá", "5,000"])

    def test_split_amount_line_reads_the_item(self):
        words = ["Tên", "hàng", "SL", "Thành", "tiền", "Bánh", "quy", "1", "25", "5.000", "Tổng", "cộng", "25,000"]
        boxes = [[10, 0, 60, 20], [70, 0, 120, 20], [130, 0, 160, 20], [170, 0, 230, 20], [240, 0, 290, 20],
                 [10, 40, 60, 60], [70, 40, 120, 60], [10, 80, 20, 100], [200, 80, 236, 100], [230, 80, 292, 100],
                 [10, 120, 60, 140], [70, 120, 120, 140], [200, 120, 290, 140]]
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["quantity"], i["amount"]) for i in res["items"]], [("Bánh quy", 1, 25000)])
        self.assertEqual(check_items(res["items"], res["summary_amounts"])["status"], "verified")


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

    def test_name_joined_to_the_registered_price_line(self):
        # train ...014bxyhq: after levelling a slanted photo the name and "Giá đăng ký" sit on one line
        words, boxes = lay(["Tên hàng SL ĐVT Đơn giá Thành tiền", "Kẹo Alpenliebe Giá đăng ký: 8,500",
                            "Giá bán: 10 GOI 8,300 83,000", "Tổng cộng: 83,000"])
        items = extract_items(words, boxes)["items"]
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"]) for i in items],
                         [("Kẹo Alpenliebe", 10, 8300, 83000)])

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
