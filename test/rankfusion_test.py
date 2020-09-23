import sys
import unittest
sys.path.append("../model")
from rankfusion import Rankfusion


class Test(unittest.TestCase):
    def setUp(self):
        self.rank = Rankfusion()

    def test_case1(self):
        als = ["0", "1", "2", "3"]
        cb = ["2", "1", "3", "0"]
        cf = ["0", "1", "4", "2"]
        algos = [als, cb, cf]
        self.rank.make_rank(algos)


if __name__ == '__main__':
    unittest.main()
