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

    def test_repeated_separator(self):
        # OCR of two overlapping crops repeats the separator: "210, ,000", "0, .460 KG" is a weight, not 460
        self.assertEqual(self.values("210, ,000 0.63 132 ,300"), [210000, 132300])
        self.assertEqual(self.values("Tổng tiền hàng: 439, ,850"), [439850])

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

    def test_weighed_goods_take_the_amount_from_the_next_line(self):
        # train ...013njyzj / ...013iijxa: the weight "0,322" is not an amount of 322; the amount is on the barcode line
        words, boxes = lay(["Mặt hàng Đơn giá SL T. Tiền", "Rau ngót 40 000/KG X 0,322 KG", "2600088003221 12 .880 12.880",
                            "Bắp cải trái tim 32 000/KG X", "0, 606 KG", "2605147006069 19 .392 19 .392",
                            "TỔNG TIỀN PHẢI T. TOÁN 32.272"])
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"], i["line_ok"]) for i in res["items"]],
                         [("Rau ngót", 0.322, 40000, 12880, True), ("Bắp cải trái tim", 0.606, 32000, 19392, True)])
        self.assertEqual(check_items(res["items"], res["summary_amounts"])["status"], "verified")

    def test_decimal_quantity_without_per_kg(self):
        # train ...013iulwh ("0.39 Kg 25,000 9,750") and ...014xyqra ("0.146 155,000 22,630" also passes as
        # 155 x 146); a weight written in the quantity column with an OCR space ("0, 52") reads the same way
        words, boxes = lay(["Tên hàng SL Đơn giá Thành tiền", "Đậu cô ve 0.39 Kg 25,000 9,750",
                            "NẠC HEO XAY 0.146 155,000 22,630", "Thịt bò thăn 0, 52 290,000 150,800",
                            "Sữa chua 2 1,500 3,000", "Tổng tiền 186,180"])
        items = extract_items(words, boxes)["items"]
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"], i["line_ok"]) for i in items],
                         [("Đậu cô ve", 0.39, 25000, 9750, True), ("NẠC HEO XAY", 0.146, 155000, 22630, True),
                          ("Thịt bò thăn", 0.52, 290000, 150800, True), ("Sữa chua", 2, 1500, 3000, True)])

    def test_one_misread_amount_repaired_by_the_printed_total(self):
        # "45 ,000 5, 000": OCR lost the "4"; 1 x 45,000 makes the items add up to the printed total
        words, boxes = lay(["Đơn giá SL Thành tiền", "Phí Ship", "12 ,000 12 000", "Trứng gà quê (Chục)",
                            "45 ,000 5, 000", "THỊT BÒ THĂN (kg)", "290 ,000 0.52 150 ,800",
                            "Tổng tiền hàng: 207, ,800"])
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["quantity"], i["unit_price"], i["amount"]) for i in res["items"]],
                         [("Phí Ship", 1, 12000, 12000), ("Trứng gà quê (Chục)", 1, 45000, 45000),
                          ("THỊT BÒ THĂN", 0.52, 290000, 150800)])
        self.assertTrue(res["items"][1]["repaired"])
        self.assertEqual(check_items(res["items"], res["summary_amounts"])["status"], "verified")

    def test_no_repair_without_a_matching_total_or_with_two_bad_lines(self):
        words, boxes = lay(["Đơn giá SL Thành tiền", "Trứng", "45 ,000 5, 000", "Tổng tiền hàng: 50,000"])
        self.assertEqual(extract_items(words, boxes)["items"][0]["amount"], 5000)
        words, boxes = lay(["Đơn giá SL Thành tiền", "Trứng", "45 ,000 5, 000", "Sữa", "30 ,000 3, 000",
                            "Tổng tiền hàng: 80,000"])
        self.assertEqual([i["amount"] for i in extract_items(words, boxes)["items"]], [5000, 3000])

    def test_garbled_header_and_invented_total_at_the_top(self):
        # train ...014wbfza: "SL" and "Đơn giá" misread, and the fine-tuned OCR wrote "Tổng thanh toán" above the store name
        words, boxes = lay(["Tổng thanh toán", "CỬA HÀNG NĂM OÁNH", "HÓA ĐƠN BÁN HÀNG", "S1. Đơn giú Thành tiền",
                            "Pin điều khiển panasonic", "6, ,000 6 000", "Xúc xích heo 25gx10", "23 ,000 23 ,000",
                            "Cộng tiền hàng 29 ,000"])
        res = extract_items(words, boxes)
        self.assertTrue(res["header_found"])
        self.assertEqual([(i["name"], i["amount"]) for i in res["items"]],
                         [("Pin điều khiển panasonic", 6000), ("Xúc xích heo 25gx10", 23000)])

    def test_gio_heo_is_a_name_and_total_quantity_is_not_a_header(self):
        # train ...013hopjk: "giò" loses its accent and reads like "giờ" (time); ...013eknsk: "Tổng số lượng hàng"
        words, boxes = lay(["Măt hàng ơn giá SL T. Tiền", "CP Bắp giò heo không xương 300g",
                            "8935058962081 51 .500 51 .500", "TỔNG TIỀN PHẢI T. TOÁN 51 .500", "Tổng số lượng hàng: 1"])
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["amount"]) for i in res["items"]], [("CP Bắp giò heo không xương 300g", 51500)])

    def test_cong_in_a_name_and_unit_only_quantity_lines(self):
        # a photographed receipt: "công nghiệp" is not "Tổng cộng", and "1 bao" / "10 quả" are not names
        words, boxes = lay(["STT TÊN HÀNG SL ĐG THÀNH TIỀN", "Gạo ST25 5kg", "bao 175 .000 175 .000",
                            "Trứng gà công nghiệp", "10 quả 32 .000 32 .000", "Rau muống", "bố 12 .000 12 .000",
                            "TỔNG TIỀN HÀNG 219 .000"])
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["amount"]) for i in res["items"]],
                         [("Gạo ST25 5kg", 175000), ("Trứng gà công nghiệp", 32000), ("Rau muống", 12000)])
        self.assertEqual(check_items(res["items"], res["summary_amounts"])["status"], "verified")

    def test_total_line_with_the_t_lost_by_ocr(self):
        # train ...014dspwj: "Tổng cộng" read as "ổng cộng"
        words, boxes = lay(["Tên hàng SL ĐVT Đơn giá Thành tiền", "hạt điều kim thanh 500g NT", "GOI 130, 000 130, 000",
                            "ổng cộng: : 130, 000", "tiền khách trả: VND 130, 000"])
        res = extract_items(words, boxes)
        self.assertEqual([(i["name"], i["amount"]) for i in res["items"]], [("hạt điều kim thanh 500g", 130000)])
        self.assertEqual(res["summary_amounts"], [130000])

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
