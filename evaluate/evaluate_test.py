import math
import unittest

from evaluate import evaluate

class evaluate_test(unittest.TestCase):
    def setUp(self):
        self.eval = evaluate()

    def test_case1(self):
        gt = [1,2,3,4,5]
        predict = [3, 1, 8, 9 ,4]
        self.eval.set_data(gt, predict)
        print("MAE : ", self.eval.MAE())
        print("MSE : ", self.eval.MSE())
        print("RMSE : ", self.eval.RMSE())
        print("AP@10 : ", self.eval.AP_k(10))
        print("NDCG : ", self.eval.NDCG())
        


if __name__ == '__main__':
    unittest.main()
