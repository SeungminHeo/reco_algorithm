from easydict import EasyDict
import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse.linalg import spsolve


class ALS:
    """Python implementation for C-ALS.

            Implementation of Collaborative Filtering for Implicit Feedback datasets.

            Reference: http://yifanhu.net/PUB/cf.pdf"""

    def __init__(self, model_config, logger):
        '''
            model_config : config.yml['model_config']['als']
            logger : getLogger('ALS')
        '''
        params = model_config['params']
        self.r_lambda = params.get('r_lambda')
        self.nf = params.get('nf')
        self.alpha = params.get('alpha')
        self.iteration = params.get('iteration')

        self.logger = logger
        self.logger.info('start ALS ')
        self.logger.info('ALS parameters -> (r_lambda : %f), (nf : %f), (alpha : %f), (iteration : %f)' % (
        self.r_lambda, self.nf, self.alpha, self.iteration))

    def set_data(self, r):  # r : sparse array(user - item)
        self.nu = r.shape[0]
        self.ni = r.shape[1]

        # initialize X and Y with very small values
        self.X = sparse.csr_matrix(np.random.rand(self.nu, self.nf) * 0.01)  # user's latent factor
        self.Y = sparse.csr_matrix(np.random.rand(self.ni, self.nf) * 0.01)  # item's latent factor

        # self.P = np.copy(r)  # Binary Rating matrix
        # self.P[self.P > 0] = 1
        # self.C = 1 + self.alpha * r  # Confidence matrix
        self.conf = self.alpha * r
        self.X_eye = sparse.eye(self.nu)
        self.Y_eye = sparse.eye(self.ni)
        self.lambda_eye = self.r_lambda * sparse.eye(self.nf)

    '''
    def _loss_function(self, predict):
        predict_error = np.square(self.P - predict)
        confidence_error = np.sum(self.C * predict_error)
        regularization = self.r_lambda * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
        total_loss = confidence_error + regularization
        return np.sum(predict_error), confidence_error, regularization, total_loss
    '''

    def _optimize_user(self):
        yT = self.Y.T
        for u in tqdm(range(self.nu)):
            conf_samp = self.conf[u, :].toarray()
            P = conf_samp.copy()
            P[P > 0] = 1
            cu = sparse.diags(conf_samp, [0]) + self.Y_eye
            yT_cu_y = yT.dot(cu).dot(self.Y)
            yT_cu_pu = yT.dot(cu).dot(P.T)
            self.X[u] = spsolve(yT_cu_y + self.lambda_eye, yT_cu_pu)

    def _optimize_item(self):
        xT = np.transpose(self.X)
        for i in range(self.ni):
            conf_samp = self.conf[:, i].T.toarray()
            P = conf_samp.copy()
            P[P > 0] = 1
            ci = sparse.diags(conf_samp, [0]) + self.X_eye
            xT_ci_x = xT.dot(ci).dot(self.X)
            xT_ci_pi = xT.dot(ci).dot(P.T)
            self.Y[i] = spsolve(xT_ci_x + self.lambda_eye, xT_ci_pi)

    def train(self):
        predict_errors = []
        confidence_errors = []
        regularization_list = []
        total_losses = []

        for i in tqdm(range(self.iteration)):
            if i != 0:
                self._optimize_user()
                self._optimize_item()
            '''
            predict = self.X.dot(self.Y.T)

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
            '''

    def predict(self, userid, id_to_idx_mapping):
        if userid not in id_to_idx_mapping.keys():  # If the ID is not included in the als model
            return -1
        idx = id_to_idx_mapping[userid]
        return self.X[idx].dot(self.Y.T)