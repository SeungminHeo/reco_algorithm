import yaml
import json
import implicit
import logging.config

from scipy import sparse
from argparse import ArgumentParser

import numpy as np
import implicit.evaluation
from sklearn.model_selection import ParameterGrid

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder, logging_time
from utils.mongo_connect import MongoConnection


class AlsFb:
    def __init__(self, fb_config, logger, featurebuilder, mongo_client):
        # ALS model
        self.max_ndcg = 0.0
        self.best_params = {}

        # convert sparse matrix
        self.user_n, self.item_n = fb_config['to_sparse']['user_n'], fb_config['to_sparse']['item_n']

        # logger
        self.logger = logger
        self.logger.info('start ALS_FB ')

        # kafka
        self.FeatureBuilder = featurebuilder
        self.kafka_config = fb_config['kafka']

        # mongo
        self.mongo_client = mongo_client

    def to_sparse_tuples(self, data, item_count=None, user_n=0, item_n=0):
        '''
        dict to sparse tuple
        user_n : Minimum number of user appearances
        item_n " Minimum number of item appearances
        '''
        if user_n > 0:
            f = lambda x: len(x[1]) > user_n
            data = dict(list(filter(f, data.items())))
        user2idx = {j: i for i, j in enumerate(data.keys())}
        items = set()
        for i in list(data.values()):
            if item_count is None:
                items = items.union(set(i))
            else:
                for item_id in i.keys():
                    if item_count[item_id] > item_n:
                        items.add(item_id)
        item2idx = {j: i for i, j in enumerate(items)}
        sparse_tuples = []
        for i in data:
            items2 = data[i]
            for j in items2:
                if j in items:
                    sparse_tuples.append((user2idx[i], item2idx[j], items2[j]))
        return sparse_tuples, user2idx, item2idx

    def to_sparse(self, data, val=None):  # sparse_tuple -> sparse.csr_matrix:
        data = np.array(data)
        if val is None:
            val = np.ones_like(data[:, 0])
        else:
            val = np.array(val)
        return sparse.csr_matrix((val, (data[:, 0], data[:, 1])),
                                 dtype="float32",
                                 shape=(data[:, 0].max() + 1, data[:, 1].max() + 1))

    @logging_time
    def train(self, time_diff_hours: int):
        item_count = {}
        cf_feature = json.loads(self.FeatureBuilder.CF(
            time_diff_hours=time_diff_hours
        ))
        grid = ParameterGrid({
            "factors": [10, 30, 50, 70, 90, 110, 150, 200, 250, 300],
            'regularization': [0.001, 0.01, .1, 1, 10, 20, 40],
            'iterations': [10, 20, 30],
            'alpha_val': [10, 50, 100]
        })

        for item in cf_feature.values():
            for item_id, count in item.items():
                item_count[item_id] = item_count.get(item_id, 0) + 1

        train_tuples, self.train_user2idx, self.train_item2idx = self.to_sparse_tuples(cf_feature, item_count,
                                                                                       self.user_n, self.item_n)
        self.train_sparse = self.to_sparse(np.array(train_tuples)[:, :-1], np.array(train_tuples)[:, -1])
        logger.debug(self.train_sparse.shape)

        for params in grid:
            model = implicit.als.AlternatingLeastSquares(factors=params['factors'],
                                                         regularization=params['regularization'],
                                                         iterations=params['iterations'])
            data_conf = (self.train_sparse.T * params['alpha_val']).astype('double')
            data_split = implicit.evaluation.train_test_split(data_conf)
            train_data, test_data = data_split[0], data_split[1]
            model.fit(train_data, show_progress=False)
            ndcg = implicit.evaluation.ndcg_at_k(model, train_user_items=train_data, test_user_items=test_data, K=5,
                                                 num_threads=8, show_progress=False)
            if self.max_ndcg < ndcg:
                self.max_ndcg = ndcg
                self.best_params = params

        logger.debug("ALS best hyperparameter : ", self.best_params)
        logger.debug("ALS max ndcg : ", self.max_ndcg)

        self.als_model = implicit.als.AlternatingLeastSquares(factors=self.best_params['factors'],
                                                              regularization=self.best_params['regularization'],
                                                              iterations=self.best_params['iterations'])
        data_conf = (self.train_sparse.T * self.best_params['alpha_val']).astype('double')
        self.als_model.fit(data_conf, show_progress=False)

    @logging_time
    def reco(self):
        train_idx2item = {y: x for x, y in self.train_item2idx.items()}
        final_reco = []

        for user_id in self.train_user2idx.keys():
            user_reco = {}
            user_idx = self.train_user2idx[user_id]
            reco_item_idx = dict(self.als_model.recommend(user_idx, self.train_sparse,
                                                          N=10, filter_already_liked_items=False))
            reco_item_id = {}
            for item_idx in reco_item_idx.keys():
                reco_item_id[train_idx2item[item_idx]] = reco_item_idx[item_idx].item()

            user_reco["piwikId"] = user_id
            user_reco['recoResult'] = reco_item_id
            final_reco.append(user_reco)

        self.mongo_client.write_many(final_reco)

    @logging_time
    def run(self, time_diff_hours: int):
        while True:
            self.train(time_diff_hours)
            self.reco()


if __name__ == "__main__":
    configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)
    fb_config = configs['main_process']['als_fb']
    logging_config = configs["logging_config"]

    # insert "logging_config" values into logging config
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('ALS_FeatureBuilder')

    # kafka parser
    parser = ArgumentParser()
    parser.add_argument("--hours", type=int, help="interval in hours from now to get train data")
    parser.add_argument("--runningEnvironment", "-re", type=str, help="environment that runs reco engine.",
                        choices=["server", "local"])

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG[args.runningEnvironment])

    mongo_client = MongoConnection('als', args.runningEnvironment)

    als_fb = AlsFb(fb_config, logger, FeatureBuilder, mongo_client)
    als_fb.run(args.hours)
