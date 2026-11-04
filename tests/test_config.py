import unittest
from pathlib import Path

from src.common.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfigs(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG_DIR)

    def test_all_files_load(self):
        self.assertEqual(set(self.cfg), {"runtime", "dataset", "model"})

    def test_keys_used_by_the_pipeline_exist(self):
        m = self.cfg["model"]
        for key in ("target_max_dim", "enhance"):
            self.assertIn(key, m["preprocessing"])
        for key in ("text_threshold", "link_threshold", "low_text", "long_size"):
            self.assertIn(key, m["detection"])
        self.assertIn("backbone", m["recognition"])
        self.assertEqual(m["layout_encoder"]["model_name"], "microsoft/layoutxlm-base")

    def test_dataset_fields(self):
        fields = self.cfg["dataset"]["mc_ocr_2021"]["annotated_fields"]
        self.assertEqual(set(fields), {"SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"})


if __name__ == "__main__":
    unittest.main()
