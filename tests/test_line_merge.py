import unittest

from src.detection.detector import TextRegion
from src.detection.line_merge import merge_segments


def box(x0, y0, x1, y1):
    return TextRegion(polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], bbox=[x0, y0, x1, y1],
                      orig_bbox=[x0 * 2, y0 * 2, x1 * 2, y1 * 2])


def boxes(regions):
    return [r.bbox for r in regions]


class MergeSegmentsTest(unittest.TestCase):
    def test_overlapping_split_number_becomes_one_segment(self):
        # "399" + "9,000" (train ...013msokr): two crops of one number that overlap
        out = merge_segments([box(528, 616, 559, 631), box(555, 617, 594, 633)])
        self.assertEqual(boxes(out), [[528, 616, 594, 633]])
        self.assertEqual(out[0].orig_bbox, [1056, 1232, 1188, 1266])
        self.assertEqual(out[0].polygon, [[528, 616], [594, 616], [594, 633], [528, 633]])

    def test_words_of_a_name_merge_but_table_columns_stay_apart(self):
        # "Thịt bò thăn" (gaps of 6 px, height 20) and the amount far to the right in its own column
        regions = [box(10, 100, 60, 120), box(66, 100, 96, 120), box(102, 101, 160, 121), box(600, 100, 690, 120)]
        self.assertEqual(boxes(merge_segments(regions)), [[10, 100, 160, 121], [600, 100, 690, 120]])

    def test_lines_are_not_merged_vertically(self):
        regions = [box(10, 100, 60, 120), box(10, 130, 60, 150)]
        self.assertEqual(boxes(merge_segments(regions)), [[10, 100, 60, 120], [10, 130, 60, 150]])

    def test_segment_width_is_capped(self):
        # twelve touching 30 px boxes of height 20 span 382 px: at most 16 x 20 = 320 px per segment
        regions = [box(10 + 32 * i, 100, 40 + 32 * i, 120) for i in range(12)]
        out = merge_segments(regions)
        self.assertGreater(len(out), 1)
        self.assertTrue(all(r.bbox[2] - r.bbox[0] <= 320 for r in out))
        self.assertEqual((out[0].bbox[0], out[-1].bbox[2]), (10, 392))

    def test_reading_order_and_small_inputs(self):
        regions = [box(300, 10, 340, 30), box(10, 200, 50, 220), box(10, 10, 50, 30)]
        self.assertEqual(boxes(merge_segments(regions)), [[10, 10, 50, 30], [300, 10, 340, 30], [10, 200, 50, 220]])
        self.assertEqual(merge_segments([]), [])
        self.assertEqual(boxes(merge_segments([box(1, 2, 30, 20)])), [[1, 2, 30, 20]])

    def test_gap_setting(self):
        regions = [box(10, 100, 60, 120), box(90, 100, 140, 120)]       # gap 30 = 1.5 heights
        self.assertEqual(len(merge_segments(regions)), 2)
        self.assertEqual(len(merge_segments(regions, max_gap=2.0)), 1)


if __name__ == "__main__":
    unittest.main()
