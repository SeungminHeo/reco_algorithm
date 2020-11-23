import yaml
import json
import implicit
import logging.config

from scipy import sparse
from argparse import ArgumentParser

import numpy as np
import implicit.evaluation
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder


class ALS_FB:
    def __init__(self, model_config, fb_config, logger, FeatureBuilder):
        ## ALS model
        self.max_ndcg = 0.0
        self.best_params = {}

        ## convert sparse matrix
        self.user_n, self.item_n = fb_config['to_sparse']['user_n'], fb_config['to_sparse']['item_n']

        ## logger
        self.logger = logger
        self.logger.info('start ALS_FB ')

        ## kafka
        self.FeatureBuilder = FeatureBuilder
        self.kafka_config = fb_config['kafka']

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

    def to_sparse(self, data, val=None): # sparse_tuple -> sparse.csr_matrix:
        data = np.array(data)
        if val is None:
            val = np.ones_like(data[:, 0])
        else:
            val = np.array(val)
        return sparse.csr_matrix((val, (data[:, 0], data[:, 1])),
                                 dtype="float32",
                                 shape=(data[:, 0].max() + 1, data[:, 1].max() + 1))

    def train(self):
        item_count = {}
        CF_feature = json.loads(self.FeatureBuilder.CF(
            group_id=CONFIG["kafka_config"]["consumer_groups"]["cf_model_feed_by_time"],
            topic_name=CONFIG["kafka_topics"]["click_log"],
            time_diff_hours=self.kafka_config.get('time')
        ))
        grid = ParameterGrid({
            "factors": [10, 30, 50, 70, 90, 110, 150, 200, 250, 300],
            'regularization': [0.001, 0.01, .1, 1, 10, 20, 40],
            'iterations': [10,20, 30],
            'alpha_val': [10, 50, 100]
        })

        for item in CF_feature.values():
            for item_id, count in item.items():
                item_count[item_id] = item_count.get(item_id, 0) + 1

        train_tuples, self.train_user2idx, self.train_item2idx = self.to_sparse_tuples(CF_feature, item_count, self.user_n, self.item_n)
        self.train_sparse = self.to_sparse(np.array(train_tuples)[:, :-1], np.array(train_tuples)[:, -1])

        print(self.train_sparse.shape) # loger 대체

        for params in tqdm(grid):
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

        print("ALS best hyperparameter : ", self.best_params)
        print("ALS max ndcg : ", self.max_ndcg)

        self.als_model = implicit.als.AlternatingLeastSquares(factors=self.best_params['factors'],
                                                         regularization=self.best_params['regularization'],
                                                         iterations=self.best_params['iterations'])
        data_conf = (self.train_sparse.T * self.best_params['alpha_val']).astype('double')
        self.als_model.fit(data_conf, show_progress=False)

    def reco(self):
        train_idx2item = {y: x for x, y in self.train_item2idx.items()}
        final_reco = []

        for user_id in self.train_user2idx.keys():
            user_reco = {}
            user_idx = self.train_user2idx[user_id]
            reco_item_idx = dict(self.als_model.recommend(user_idx, self.train_sparse, N=20, filter_already_liked_items=False))
            reco_item_id = {}
            for item_idx in reco_item_idx.keys():
                reco_item_id[train_idx2item[item_idx]] = reco_item_idx[item_idx]

            user_reco["piwikId"] = user_id
            user_reco['recoResult'] = reco_item_id
            final_reco.append(user_reco)

        print(final_reco)


            
    def run(self):
        while True:
            self.train()
            self.reco()



if __name__ == "__main__":
    configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)
    model_config = configs["model_config"]['implicit_als']
    fb_config = configs['main_process']['als_fb']
    logging_config = configs["logging_config"]

    # insert "logging_config" values into logging config
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('ALS feature_builder')

    # kafka parser
    parser = ArgumentParser()
    parser.add_argument("--numMessage", "-n", type=int, help="number of messages to poll")

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG)

    als_fb = ALS_FB(model_config, fb_config, logger, FeatureBuilder)
    als_fb.run()