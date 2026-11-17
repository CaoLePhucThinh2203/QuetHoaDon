import unittest

from src.evaluation import app_metrics
from src.extraction.datetime_rules import (date_at, datetime_candidates, extract_time, improve_timestamps,
                                           read_datetime)


class TestExtractTime(unittest.TestCase):
    def test_samples(self):
        self.assertEqual(extract_time("Giờ: 16:49"), "16:49")
        self.assertEqual(extract_time("17h23"), "17:23")
        self.assertEqual(extract_time("22/01/2020 Giờ: 16:49"), "16:49")
        self.assertEqual(extract_time("11/08/2020-08:06"), "08:06")
        self.assertEqual(extract_time("14/08/2 2020 08:06"), "08:06")
        self.assertEqual(extract_time("12:29:42"), "12:29")
        self.assertIsNone(extract_time("22/01/2020"))
        self.assertIsNone(extract_time("25:99"))
        self.assertIsNone(extract_time(None))

    def test_same_as_the_scoring_metric(self):
        for text in ("Giờ: 16:49", "17h23", "11/08/2020-08:06", "14/08/2 2020 08:06", "22/01/2020", "", "9:05 pm"):
            self.assertEqual(extract_time(text), app_metrics.extract_time(text), text)


class TestDateAt(unittest.TestCase):
    def test_single_word(self):
        self.assertEqual(date_at(["11/08/2020"], 0), ("11/08/2020", 1))
        self.assertEqual(date_at(["x", "09/08/20"], 1), ("09/08/2020", 2))

    def test_split_year_digit(self):
        self.assertEqual(date_at(["14/08/2", "2020", "x"], 0), ("14/08/2020", 2))

    def test_day_month_then_year_word(self):
        self.assertEqual(date_at(["20/09", "2019"], 0), ("20/09/2019", 2))
        self.assertEqual(date_at(["20/09", "x", "2019"], 0), ("20/09/2019", 3))
        self.assertEqual(date_at(["bán:08/08", "2020"], 0), ("08/08/2020", 2))

    def test_rejects_things_that_are_not_dates(self):
        self.assertIsNone(date_at(["12.50", "2020"], 0))            # month 50
        self.assertIsNone(date_at(["5.000"], 0))
        self.assertIsNone(date_at(["20/09", "x", "y", "2019"], 0))  # year too far away
        self.assertIsNone(date_at(["20/09", "1999"], 0))            # not a plausible receipt year
        self.assertIsNone(date_at(["Tong"], 0))

    def test_does_not_swallow_a_following_date(self):
        self.assertEqual(date_at(["x", "11/08/2020"], 0), None)
        self.assertEqual(date_at(["11/08/2020", "12/08/2020"], 0), ("11/08/2020", 1))


class TestReadDatetime(unittest.TestCase):
    def test_time_after_the_date(self):
        self.assertEqual(read_datetime(["Ngay", "11/08/2020", "20:50", "Tong"]), "11/08/2020 20:50")

    def test_time_glued_to_the_date(self):
        self.assertEqual(read_datetime(["09/08/2020-17:23"]), "09/08/2020 17:23")

    def test_time_before_the_date(self):
        self.assertEqual(read_datetime(["Gio", "20:50", "11/08/2020"]), "11/08/2020 20:50")

    def test_h_separator(self):
        self.assertEqual(read_datetime(["11/08/2020", "20h05"]), "11/08/2020 20:05")

    def test_no_time(self):
        self.assertEqual(read_datetime(["11/08/2020", "Tong", "tien"]), "11/08/2020")

    def test_time_too_far_is_ignored(self):
        words = ["11/08/2020"] + ["a"] * 6 + ["20:50"]
        self.assertEqual(read_datetime(words), "11/08/2020")

    def test_time_after_a_split_date(self):
        self.assertEqual(read_datetime(["20/09", "2019", "12:29:42"]), "20/09/2019 12:29")

    def test_first_date_wins_and_no_date(self):
        self.assertEqual(read_datetime(["11/08/2020", "08:06", "12/08/2020", "09:09"]), "11/08/2020 08:06")
        self.assertIsNone(read_datetime(["Tong", "74,000"]))
        self.assertIsNone(read_datetime([]))

    def test_candidates_cover_every_date(self):
        c = datetime_candidates(["11/08/2020", "08:06", "x", "12/08/2020"])
        self.assertEqual([(x["date"], x["time"]) for x in c], [("11/08/2020", "08:06"), ("12/08/2020", None)])


class TestImproveTimestamps(unittest.TestCase):
    def test_adds_the_time_and_keeps_receipts_without_a_date(self):
        records = [{"img_id": "a", "words": ["Ngay", "11/08/2020", "20:50"]},
                   {"img_id": "b", "words": ["Tong", "74,000"]},
                   {"img_id": "c", "words": ["20/09", "2019"]}]
        preds = {"a": {"TIMESTAMP": "11/08/2020", "SELLER": "X"},
                 "b": {"TIMESTAMP": "old value"},
                 "c": {"TIMESTAMP": None}}
        out = improve_timestamps(records, preds)
        self.assertEqual(out["a"], {"TIMESTAMP": "11/08/2020 20:50", "SELLER": "X"})
        self.assertEqual(out["b"]["TIMESTAMP"], "old value")
        self.assertEqual(out["c"]["TIMESTAMP"], "20/09/2019")
        self.assertEqual(preds["a"]["TIMESTAMP"], "11/08/2020")          # input is not modified


if __name__ == "__main__":
    unittest.main()
