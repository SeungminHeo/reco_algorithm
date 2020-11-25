import logging.config
import time
from argparse import ArgumentParser

import tensorflow as tf
import yaml

from model.vae import VAE, VAE_STRT
from utils.fb_utils import *
from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder
from utils.mongo_connect import MongoConnection


class VAE_FB:
    def __init__(self, model_config, fb_config, logger, FeatureBuilder=None, mongo_client=None):
        self.logger = logger
        self.logger.info("Begin VAE_FB")

        self.model_config = model_config
        self.FeatureBuilder = FeatureBuilder
        self.kafka_config = fb_config["kafka"]

        self.mongo_client = mongo_client

        layers_common = {"activity_regularizer": tf.keras.regularizers.l2(0.0),
                         "bias_initializer": tf.keras.initializers.TruncatedNormal(stddev=0.001)}

        layer_1 = {"units": 60,
                   "activation": "tanh",
                   **layers_common,
                   }

        layer_2 = {"units": 20,
                   **layers_common,
                   }

        self.SPECS = [layer_1, layer_2]

        self.additional_specs = {"dropout": 0.5,
                                 "normalize_eps": True,
                                 }

    def train(self, time_diff_hours):

        CF_feature = json.loads(self.FeatureBuilder.CF(
            time_diff_hours=time_diff_hours
        ))

        train_items = set()
        for key, val in CF_feature.items():
            train_items.update(set(val.keys()))

        self.train_item2idx = {i: j for j, i in enumerate(train_items)}
        self.train_user2idx = {i: j for j, i in enumerate(CF_feature.keys())}
        sparse_tuples, _, _ = to_sparse_tuples(CF_feature, self.train_user2idx, self.train_item2idx)

        ui_matrix = np.array(sparse_tuples).astype(np.float32)[:, :-1]
        ui_rel = np.array(sparse_tuples)[:, -1]

        self.ui_sparse = to_sparse(ui_matrix, ui_rel)

        user_size, item_size = self.ui_sparse.shape[0], self.ui_sparse.shape[1]
        self.SPECS[0]["input_dim"] = item_size

        self.logger.info("Begin Training...")

        self.model = VAE(optimizer=tf.keras.optimizers.Adam,
                         enc_specs=self.SPECS,
                         additional_specs=self.additional_specs,
                         model_config=self.model_config,
                         logger=self.logger,
                         anneal=1,
                         model=VAE_STRT)

        self.stime = time.time()

        self.model.set_data(self.ui_sparse, user_size, item_size)

        trained_model = self.model.train(False)
        return trained_model

    def reco(self):
        train_idx2item = {i: j for j, i in self.train_item2idx.items()}
        train_idx2user = {i: j for j, i in self.train_user2idx.items()}

        rec_pool, rec_logits = self.model.recommend(inp=self.ui_sparse.toarray().astype(np.float32),
                                                    idx2item=train_idx2item,
                                                    idx2user=train_idx2user,
                                                    topk=5,
                                                    logits=True)

        self.logger.info("Recommendation Successful!")
        self.logger.info("Time taken: {:.3f} SECONDS".format((time.time() - self.stime) / 60))

        final_reco = []

        for user, item in rec_pool.items():
            user_reco = dict()
            user_reco["piwikId"] = user
            user_reco['recoResult'] = item
            final_reco.append(user_reco)

        self.mongo_client.write_many(final_reco)

    def run(self, time_diff_hours: int):
        while True:
            self.train(time_diff_hours)
            self.reco()


if __name__ == "__main__":
    configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)
    model_config = configs["model_config"]["vae"]
    fb_config = configs["main_process"]["vae_fb"]
    logging_config = configs["logging_config"]

    # insert "logging_config" values into logging config
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('VAE_FeatureBuilder')

    parser = ArgumentParser()
    parser.add_argument("--hours", type=int, help="interval in hours from now to get train data")
    parser.add_argument("--runningEnvironment", "-re", type=str, help="environment that runs reco engine.",
                        choices=["server", "local"])

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG[args.runningEnvironment])

    mongo_client = MongoConnection('vae', args.runningEnvironment)

    vae_fb = VAE_FB(model_config, fb_config, logger, FeatureBuilder, mongo_client)
    vae_fb.run(args.hours)
