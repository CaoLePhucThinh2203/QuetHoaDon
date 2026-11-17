import unittest

from src.training.data import word_token_counts


class FakeBackend:
    def __init__(self):
        self.padding = True

    def no_padding(self):
        self.padding = False

    def no_truncation(self):
        pass


class FakeTokenizer:
    """Pads every encoding to 512 tokens while the backend still has padding switched on."""

    def __init__(self):
        self._tokenizer = FakeBackend()

    def tokenize(self, word):
        return ["x"] * (512 if self._tokenizer.padding else max(1, len(word) // 3))


class WordTokenCountsTest(unittest.TestCase):
    def test_padding_left_by_a_processor_call_does_not_inflate_the_counts(self):
        tok = FakeTokenizer()
        self.assertEqual(word_token_counts(tok, ["abcdef", "ab"]), [2, 1])

    def test_tokenizer_without_backend_still_works(self):
        class Plain:
            def tokenize(self, w):
                return list(w)
        self.assertEqual(word_token_counts(Plain(), ["abc", "d"]), [3, 1])


if __name__ == "__main__":
    unittest.main()
