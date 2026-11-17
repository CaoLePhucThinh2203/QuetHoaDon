import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_strict", Path(__file__).resolve().parent.parent / "scripts" / "compare_strict.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


class McNemarTest(unittest.TestCase):
    def test_no_discordant_pairs(self):
        self.assertEqual(cs.mcnemar_exact(0, 0), 1.0)

    def test_balanced_is_not_significant(self):
        self.assertEqual(cs.mcnemar_exact(5, 5), 1.0)

    def test_known_value(self):
        self.assertAlmostEqual(cs.mcnemar_exact(10, 0), 2 / 1024)      # 2 * P(X = 0), X ~ Bin(10, 0.5)

    def test_symmetric(self):
        self.assertAlmostEqual(cs.mcnemar_exact(18, 2), cs.mcnemar_exact(2, 18))


class BootstrapTest(unittest.TestCase):
    def test_interval_contains_the_mean_difference(self):
        pairs = [(0, 1)] * 30 + [(1, 1)] * 40 + [(0, 0)] * 30
        low, high = cs.bootstrap_diff(pairs, n_boot=500)
        self.assertLess(low, 0.30)
        self.assertGreater(high, 0.30)
        self.assertGreater(low, 0.0)


if __name__ == "__main__":
    unittest.main()
