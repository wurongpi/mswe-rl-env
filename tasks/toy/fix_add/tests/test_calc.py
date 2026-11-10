import os, sys, unittest
sys.path.insert(0, os.environ.get("WORKSPACE", "/workspace"))
from calc import add, mul


class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

    def test_mul_untouched(self):
        self.assertEqual(mul(3, 4), 12)


if __name__ == "__main__":
    unittest.main()
