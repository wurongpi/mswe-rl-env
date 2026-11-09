import os, sys, unittest
sys.path.insert(0, os.environ.get("WORKSPACE", "/workspace"))
from fizzbuzz import fizzbuzz


class T(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(fizzbuzz(15)), 15)

    def test_values(self):
        self.assertEqual(fizzbuzz(5), ["1", "2", "Fizz", "4", "Buzz"])
        self.assertEqual(fizzbuzz(15)[-1], "FizzBuzz")


if __name__ == "__main__":
    unittest.main()
