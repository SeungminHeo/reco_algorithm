import unittest
import sys
import numpy as np
sys.path.append("../model")
from als import ALS


class Test(unittest.TestCase):
    def setUp(self):
        self.als = ALS()

    def test_case1(self):
        r = np.array([[0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
                      [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
                      [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                      [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
                      [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                      [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0]])
        self.als.set_data(r)
        predict = self.als.train()

    def test_case2(self):
        r = np.array([[0, 0, 1],
                      [0, 0, 0],
                      [0, 3, 4],
                      [5, 8, 9]])
        self.als.set_data(r)
        predict = self.als.train()
        

if __name__ == '__main__':
    unittest.main()
