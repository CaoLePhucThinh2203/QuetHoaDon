import os
import tempfile
import unittest

import cv2
import numpy as np

from src.preprocessing.image_preprocessor import ImagePreprocessor


def text_page(rotation=0.0):
    img = np.full((900, 600, 3), 255, np.uint8)
    for i in range(25):
        cv2.putText(img, f"Dong hang so {i} 123456 abcdef", (30, 40 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if rotation:
        m = cv2.getRotationMatrix2D((300, 450), rotation, 1.0)
        img = cv2.warpAffine(img, m, (600, 900), borderValue=(255, 255, 255))
    return img


class TestSkew(unittest.TestCase):
    def test_estimate_skew_recovers_rotation(self):
        pre = ImagePreprocessor(enhance=False)
        for true_angle in (4, -7, 12):
            gray = cv2.cvtColor(text_page(true_angle), cv2.COLOR_BGR2GRAY)
            self.assertAlmostEqual(pre.estimate_skew(gray), -true_angle, delta=0.75)

    def test_straight_page_not_rotated(self):
        pre = ImagePreprocessor(enhance=False)
        self.assertEqual(pre.estimate_skew(cv2.cvtColor(text_page(0), cv2.COLOR_BGR2GRAY)), 0.0)


class TestProcess(unittest.TestCase):
    def test_resize_and_scale(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big.jpg")
            cv2.imwrite(path, np.full((2000, 1000, 3), 255, np.uint8))
            res = ImagePreprocessor(target_max_dim=1000, enhance=False).process(path)
            self.assertEqual(max(res.image.shape[:2]), 1000)
            self.assertAlmostEqual(res.scale, 0.5)
            self.assertEqual(res.original.shape[:2], (2000, 1000))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ImagePreprocessor().process("/nonexistent/x.jpg")


if __name__ == "__main__":
    unittest.main()
