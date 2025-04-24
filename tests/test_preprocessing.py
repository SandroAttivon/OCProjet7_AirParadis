import unittest
from utils import clean_text

class PreprocessingTestCase(unittest.TestCase):
    def test_clean_text_basic(self):
        text = "I love @airparadis! Visit us: https://t.co/example"
        cleaned = clean_text(text)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("!", cleaned)

    def test_clean_text_stopwords(self):
        text = "This is a test with some stopwords"
        cleaned = clean_text(text)
        for stopword in ["is", "a", "with", "some"]:
            self.assertNotIn(stopword, cleaned)

if __name__ == '__main__':
    unittest.main()
