import math


class Evaluate:
    '''
    truth = [2, 4, 3, 5, 1]
    predict = [3, 4, 5, 3, 2]
    Evaluate the recommendation system.
    '''

    def __init__(self, logger):
        self.logger = logger
        self.logger.info('start evaluate')

    def set_data(self, truth, predict):
        self.truth = truth
        self.predict = predict

    # MAE
    def mae(self):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.predict):
            sum_rating += abs(p - r)
        self.logger.info('MAE : %f' % (sum_rating / num_rating))
        return sum_rating / num_rating

    # MSE
    def mse(self):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값의 제곱 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.predict):
            sum_rating += abs(p - r) ** 2
        self.logger.info('MSE : %f' % (sum_rating / num_rating))
        return sum_rating / num_rating

    # RMSE
    def rmse(self):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값의 제곱 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.predict):
            sum_rating += abs(p - r) ** 2
        self.logger.info('RMSE : %f' % ((sum_rating / num_rating) ** (1 / 2)))
        return (sum_rating / num_rating) ** (1 / 2)

    # AP@k
    def ap_k(self, k):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        ap_sum = 0
        true_count = 0
        m = len(self.predict)
        for idx in range(m):
            k = k - 1
            if self.predict[idx] in self.truth:
                true_count += 1
                ap_sum += true_count / (idx + 1)
            if k == 0:
                break
        self.logger.info('AP@%f : %f' % (k, ap_sum / m))
        return ap_sum / m

    # DCG
    def dcg(self):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        dcg = 0.0
        for i, r in enumerate(self.predict):
            if r in self.truth:
                dcg += 1.0 / math.log(i + 2, 2)
        self.logger.info('DCG : %f' % (dcg))
        return dcg

    # NDCG
    def ndcg(self):
        if not hasattr(self, "truth") or not hasattr(self, "predict"):
            self.logger.error("don't set dataset")
            return -1
        idcg = 0.0
        len_truth = len(self.truth)
        for i in range(len_truth):
            idcg += 1.0 / math.log(i + 2, 2)

        dcg = 0.0
        for i, r in enumerate(self.predict):
            if r in self.truth:
                dcg += 1.0 / math.log(i + 2, 2)
        self.logger.info('NDCG : %f' % (dcg / idcg))
        return dcg / idcg
