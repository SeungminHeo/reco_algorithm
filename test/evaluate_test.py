import math
import unittest

import sys
sys.path.append("../evaluate")

from evaluate import Evaluate


class evaluate_test(unittest.TestCase):
    def setUp(self):
        self.eval = evaluate()

    def test_case1(self):
        gt = [1, 2, 3, 4, 5]
        predict = [3, 1, 8, 9, 4]
        self.eval.set_data(gt, predict)
        self.eval.MAE()
        self.eval.MSE()
        self.eval.RMSE()
        self.eval.AP_k(10)
        self.eval.NDCG()

    def test_case2(self):
        self.eval.NDCG()


if __name__ == '__main__':
    unittest.main()
