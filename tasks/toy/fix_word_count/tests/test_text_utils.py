import os, sys, unittest
sys.path.insert(0, os.environ.get("WORKSPACE", "/workspace"))
from text_utils import word_count


class T(unittest.TestCase):
    def test_whitespace(self):
        self.assertEqual(word_count("a  b\tc"), 3)
        self.assertEqual(word_count("  hello world \n"), 2)

    def test_empty(self):
        self.assertEqual(word_count(""), 0)


if __name__ == "__main__":
    unittest.main()
