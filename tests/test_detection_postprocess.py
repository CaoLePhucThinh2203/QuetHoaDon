import unittest

from src.detection.detector import TextRegion
from src.detection.postprocess import drop_nested, sort_reading_order


def region(x0, y0, x1, y1, tag=None):
    r = TextRegion(polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], bbox=[x0, y0, x1, y1], orig_bbox=[x0, y0, x1, y1])
    r.tag = tag
    return r


class TestDropNested(unittest.TestCase):
    def test_comma_box_inside_amount_is_removed(self):
        amount = region(315, 262, 392, 285, "27,000")
        comma = region(345, 276, 352, 287, "comma")      # ~90% inside the amount box
        neighbour = region(440, 262, 450, 285, "n")
        kept = drop_nested([amount, comma, neighbour])
        self.assertEqual([r.tag for r in kept], ["27,000", "n"])

    def test_identical_duplicates_keep_one(self):
        kept = drop_nested([region(0, 0, 50, 20, "a"), region(0, 0, 50, 20, "b")])
        self.assertEqual(len(kept), 1)

    def test_empty(self):
        self.assertEqual(drop_nested([]), [])


class TestReadingOrder(unittest.TestCase):
    def test_rows_then_columns(self):
        import random
        rng = random.Random(0)
        regs = []
        for row in range(6):
            for col in range(4):
                y = 50 + row * 40 + rng.randint(-4, 4)
                regs.append(region(20 + col * 100, y, 100 + col * 100, y + 20, (row, col)))
        shuffled = regs[:]
        rng.shuffle(shuffled)
        order = [r.tag for r in sort_reading_order(shuffled)]
        self.assertEqual(order, [(r, c) for r in range(6) for c in range(4)])

    def test_empty(self):
        self.assertEqual(sort_reading_order([]), [])


if __name__ == "__main__":
    unittest.main()
