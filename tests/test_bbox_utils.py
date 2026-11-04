"""Unit tests for geometry and bounding box manipulation utilities."""
import unittest

from src.common.bbox_utils import xywh_to_xyxy, xyxy_to_xywh, polygon_to_xyxy, contains_ratio
from src.common.bbox_utils import (
    compute_iou,
    denormalize_bbox,
    normalize_bbox,
    polygon_to_bbox,
)


class TestBboxUtils(unittest.TestCase):

    def test_polygon_to_bbox_standard(self):
        # 4 vertices: (10, 20), (100, 20), (100, 80), (10, 80)
        seg = [10, 20, 100, 20, 100, 80, 10, 80]
        bbox = polygon_to_bbox(seg)
        self.assertEqual(bbox, [10, 20, 90, 60])

    def test_polygon_to_bbox_nested(self):
        seg = [[10, 20, 100, 20, 100, 80, 10, 80]]
        bbox = polygon_to_bbox(seg)
        self.assertEqual(bbox, [10, 20, 90, 60])

    def test_normalize_bbox_xywh(self):
        # Image width 1000, height 500
        # bbox: x=100, y=50, w=200, h=100
        bbox = [100, 50, 200, 100]
        norm = normalize_bbox(bbox, width=1000, height=500, scale=1000, is_xywh=True)
        # x0 = 100/1000 * 1000 = 100
        # y0 = 50/500 * 1000 = 100
        # x1 = 300/1000 * 1000 = 300
        # y1 = 150/500 * 1000 = 300
        self.assertEqual(norm, [100, 100, 300, 300])

    def test_normalize_bbox_clamping_and_invariants(self):
        # Coordinates out of bounds should be clamped to [0, 1000]
        bbox = [-50, -20, 1200, 2500]
        norm = normalize_bbox(bbox, width=1000, height=2000, scale=1000, is_xywh=True)
        self.assertTrue(0 <= norm[0] <= norm[2] <= 1000)
        self.assertTrue(0 <= norm[1] <= norm[3] <= 1000)
        self.assertEqual(norm[0], 0)
        self.assertEqual(norm[1], 0)
        self.assertEqual(norm[2], 1000)
        self.assertEqual(norm[3], 1000)

    def test_denormalize_bbox(self):
        norm = [100, 100, 300, 300]
        orig = denormalize_bbox(norm, width=1000, height=500, scale=1000)
        self.assertEqual(orig, [100, 50, 300, 150])

    def test_compute_iou_identical(self):
        boxA = [10, 10, 50, 50]
        boxB = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

    def test_compute_iou_disjoint(self):
        boxA = [0, 0, 10, 10]
        boxB = [20, 20, 30, 30]
        self.assertEqual(compute_iou(boxA, boxB), 0.0)

    def test_compute_iou_partial(self):
        boxA = [0, 0, 20, 20]  # area = 400
        boxB = [10, 0, 30, 20]  # area = 400, intersection = 10*20 = 200, union = 600
        self.assertAlmostEqual(compute_iou(boxA, boxB), 200.0 / 600.0)



class TestBoxFormats(unittest.TestCase):
    def test_xywh_xyxy_roundtrip(self):
        self.assertEqual(xywh_to_xyxy([10, 20, 30, 40]), [10, 20, 40, 60])
        self.assertEqual(xyxy_to_xywh([10, 20, 40, 60]), [10, 20, 30, 40])

    def test_polygon_to_xyxy(self):
        self.assertEqual(polygon_to_xyxy([0, 0, 10, 0, 10, 5, 0, 5]), [0, 0, 10, 5])

    def test_contains_ratio(self):
        self.assertAlmostEqual(contains_ratio([2, 2, 4, 4], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(contains_ratio([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)
        self.assertEqual(contains_ratio([0, 0, 1, 1], [5, 5, 6, 6]), 0.0)


if __name__ == "__main__":
    unittest.main()
