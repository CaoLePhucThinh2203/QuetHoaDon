import os
import tempfile
import unittest

from src.evaluation.latency import dir_size_mb, format_latency_table, mean_stage_seconds, summarize_times


class TestLatency(unittest.TestCase):
    def test_summarize_times(self):
        s = summarize_times([0.1, 0.2, 0.3])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean_ms"], 200.0)
        self.assertAlmostEqual(s["median_ms"], 200.0)
        self.assertAlmostEqual(s["p95_ms"], 290.0)
        self.assertAlmostEqual(s["min_ms"], 100.0)
        self.assertAlmostEqual(s["max_ms"], 300.0)

    def test_summarize_empty(self):
        s = summarize_times([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["mean_ms"], 0.0)

    def test_dir_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "sub"))
            for rel, size in (("a.bin", 1000), (os.path.join("sub", "b.bin"), 500)):
                with open(os.path.join(tmp, rel), "wb") as f:
                    f.write(b"x" * size)
            self.assertAlmostEqual(dir_size_mb(tmp), 0.0015)
            self.assertAlmostEqual(dir_size_mb(os.path.join(tmp, "a.bin")), 0.001)

    def test_mean_stage_seconds(self):
        m = mean_stage_seconds([{"ocr": 2.0, "detect": 1.0}, {"ocr": 4.0}, None])
        self.assertAlmostEqual(m["ocr"], 3.0)
        self.assertAlmostEqual(m["detect"], 1.0)

    def test_table(self):
        row = {"system": "M1b", "device": "cuda", "n": 60, "mean_ms": 33.0, "median_ms": 32.0, "p95_ms": 40.0,
               "parameters": 368658377, "size_mb": 1474.0}
        table = format_latency_table([row])
        self.assertIn("| M1b | cuda | 60 | 33.0 | 32.0 | 40.0 | 368,658,377 | 1474 |", table)


if __name__ == "__main__":
    unittest.main()
