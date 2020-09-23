import logging
import logging.config
import yaml
from easydict import EasyDict
import numpy as np
logging.config.fileConfig('../conf/logging.conf')


class ALS:
    def __init__(self):
        """Python implementation for C-ALS.

        Implementation of Collaborative Filtering for Implicit Feedback datasets.

        Reference: http://yifanhu.net/PUB/cf.pdf"""
        params = yaml.load(open("../conf/model.yml").read(), Loader=yaml.Loader)['model']['als']['params']
        self.r_lambda = params.get('r_lambda')
        self.nf = params.get('nf')
        self.alpha = params.get('alpha')
        self.iteration = params.get('iteration')

        self.logger = logging.getLogger('ALS')
        self.logger.info('start ALS ')
        self.logger.info('ALS parameters -> (r_lambda : %f), (nf : %f), (alpha : %f), (iteration : %f)' %(self.r_lambda, self.nf, self.alpha, self.iteration))

    def set_data(self, r):  # R : array(user - item)
        self.nu = r.shape[0]
        self.ni = r.shape[1]

        # initialize X and Y with very small values
        self.X = np.random.rand(self.nu, self.nf) * 0.01  # user's latent factor
        self.Y = np.random.rand(self.ni, self.nf) * 0.01  # item's latent factor

        self.P = np.copy(r)  # Binary Rating matrix
        self.P[self.P > 0] = 1
        self.C = 1 + self.alpha * r  # Confidence matrix

    def _loss_function(self, predict):  # xTy : predict
        predict_error = np.square(self.P - predict)
        confidence_error = np.sum(self.C * predict_error)
        regularization = self.r_lambda * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
        total_loss = confidence_error + regularization
        return np.sum(predict_error), confidence_error, regularization, total_loss

    def _optimize_user(self):
        yt = np.transpose(self.Y)
        for u in range(self.nu):
            cu = np.diag(self.C[u])
            yt_cu_y = np.matmul(np.matmul(yt, cu), self.Y)
            li = np.dot(self.r_lambda, np.identity(self.nf))
            yt_cu_pu = np.matmul(np.matmul(yt, cu), self.P[u])
            self.X[u] = np.linalg.solve(yt_cu_y + li, yt_cu_pu)

    def _optimize_item(self):
        xt = np.transpose(self.X)
        for i in range(self.ni):
            ci = np.diag(self.C[:, i])
            xt_ci_x = np.matmul(np.matmul(xt, ci), self.X)
            li = np.dot(self.r_lambda, np.identity(self.nf))
            xt_ci_pi = np.matmul(np.matmul(xt, ci), self.P[:, i])
            self.Y[i] = np.linalg.solve(xt_ci_x + li, xt_ci_pi)

    def train(self):
        predict_errors = []
        confidence_errors = []
        regularization_list = []
        total_losses = []

        for i in range(self.iteration):
            if i != 0:
                self._optimize_user()
                self._optimize_item()
            predict = np.matmul(self.X, np.transpose(self.Y))
            predict_error, confidence_error, regularization, total_loss = self._loss_function(predict)

            predict_errors.append(predict_error)
            confidence_errors.append(confidence_error)
            regularization_list.append(regularization)
            total_losses.append(total_loss)

            self.logger.debug('----------------step %d----------------' % i)
            self.logger.debug("predict error: %f" % predict_error)
            self.logger.debug("confidence error: %f" % confidence_error)
            self.logger.debug("regularization: %f" % regularization)
            self.logger.debug("total loss: %f" % total_loss)

        predict = np.matmul(self.X, np.transpose(self.Y))
        return predict
