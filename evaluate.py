import math
import logging
import logging.config

logging.config.fileConfig('./conf/logging.conf')

class evaluate:
    '''
    ground_truth = [2, 4, 3, 5, 1]
    prediction = [3, 4, 5, 3, 2]
    Evaluate the recommendation system.
    '''
    def __init__(self):
        self.logger = logging.getLogger('evaluate')
        self.logger.info('start evaluate')

    def set_data(self, truth, predict):
        self.truth = truth
        self.predict = predict

    # MAE
    def MAE(self):
        self.logger.info('MAE')
        if self.truth is None or self.predict is None:
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.prediction):
            sum_rating += abs(p - r)
        return sum_rating / num_rating

    # MSE
    def MSE(self):
        self.logger.info('MSE')
        if self.truth is None or self.predict is None:
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값의 제곱 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.prediction):
            sum_rating += abs(p - r) ** 2
        return sum_rating / num_rating

    # RMSE
    def RMSE(self):
        self.logger.info('RMSE')
        if self.truth is None or self.predict is None:
            self.logger.error("don't set dataset")
            return -1
        sum_rating = 0  # P-R 절대값의 제곱 합
        num_rating = len(self.truth)
        for r, p in zip(self.truth, self.prediction):
            sum_rating += abs(p - r) ** 2
        return (sum_rating / num_rating) ** (1 / 2)

    # AP@k
    def AP_k(self, k):
        self.logger.info('AP@K')
        if self.truth is None or self.predict is None:
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
        return ap_sum / m

    # DCG
    def DCG(self):
        self.logger.info('DCG')
        if self.truth is None or self.predict is None:
            self.logger.error("don't set dataset")
            return -1
        dcg = 0.0
        for i, r in enumerate(self.predict):
            if r in self.truth:
                dcg += 1.0 / math.log(i + 2, 2)
        return dcg

    # NDCG
    def NDCG(self):
        self.logger.info('NDCG')
        if self.truth is None or self.predict is None:
            self.logger.error("don't set dataset")
            return -1
        idcg = 0.0
        len_truth = len(self.truth)
        for i in range(len_truth):
            idcg += 1.0 / math.log(i + 2, 2)
        return self.DCG()/idcg
