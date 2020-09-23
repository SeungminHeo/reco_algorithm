import unittest
import sys
sys.path.append("../evaluate")

from evaluate import Evaluate


class Test(unittest.TestCase):
    def setUp(self):
        self.eval = Evaluate()

    def test_case1(self):
        gt = [1, 2, 3, 4, 5]
        predict = [3, 1, 8, 9, 4]
        self.eval.set_data(gt, predict)
        self.eval.mae()
        self.eval.mse()
        self.eval.rmse()
        self.eval.ap_k(10)
        self.eval.ndcg()

    def test_case2(self):
        self.eval.ndcg()


if __name__ == '__main__':
    unittest.main()
