import sys
import copy
from typing import List
import yaml
import numpy as np
import tensorflow as tf
from dataloader import *
from vae import VAE, VAE_STRT
sys.path.insert(1, '../evaluate')
from evaluate import Evaluate

class VAE_FB:
    def __init__(self, model_config, fb_config, logger, FeatureBuilder=None):
        self.logger = logger
        self.logger.info("Begin VAE_FB")

        self.model_config = model_config
        self.FeatureBuilder = FeatureBuilder
        self.kafka_config = fb_config["kafka"]

        layers_common = {"activity_regularizer": tf.keras.regularizers.l2(0.0), 
                         "bias_initializer": tf.keras.initializers.TruncatedNormal(stddev=0.001)}
            
        layer_1 = {"units": 60, 
                   "activation": "tanh", 
                   #"input_dim": item_size,
                   **layers_common,
                   }

        layer_2 = {"units": 20,
                   **layers_common, 
                   }

        self.SPECS = [layer_1, layer_2]

        self.additional_specs = {"dropout": 0.5,
                                 "normalize_eps": True,
                                 }

    def train(self):
        
        CF_feature = json.loads(self.FeatureBuilder.CF(
            group_id=CONFIG["kafka_config"]["consumer_groups"]["cf_model_feed_by_time"],
            topic_name=CONFIG["kafka_topics"]["click_log"],
            time_diff_hours=self.kafka_config.get('time')
        ))

        train_items = set()
        for key, val in CF_feature.items():
            train_items.update(set(val.keys()))

        self.train_item2idx = {i:j for j,i in enumerate(train_items)}
        self.train_user2idx = {i:j for j,i in enumerate(CF_feature.keys())}
        sparse_tuples, _, _ = to_sparse_tuples(CF_feature, self.train_user2idx, self.train_item2idx)

        ui_matrix = np.array(sparse_tuples).astype(np.float32)[:,:-1]
        ui_rel = np.array(sparse_tuples)[:,-1]

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
        train_idx2item = {i:j for j,i in self.train_item2idx.items()}
        train_idx2user = {i:j for j,i in self.train_user2idx.items()}

        rec_pool, rec_logits = self.model.recommend(inp=self.ui_sparse.toarray().astype(np.float32),
                                                    idx2item=train_idx2item,
                                                    idx2user=train_idx2user,
                                                    topk=5,
                                                    logits=True)

        self.logger.info("Recommendation Successful!")
        self.logger.info("Time taken: {:.3f} SECONDS".format((time.time()-self.stime)/60))

        with open(config.rec_fname, "w") as f:
            json.dump(rec_pool, f)

        self.logger.info(f"Recommendation saved to {config.rec_fname}")
    
    def run(self):
        self.train()
        self.reco()

if __name__=="__main__":
    configs = yaml.load(open("./config.yml").read(), Loader=yaml.Loader)
    model_config = configs["model_config"]["vae"]
    fb_config = configs["main_process"]["vae_fb"]
    logging_config = configs["logging_config"]

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('evaluate')

    parser = ArgumentParser()
    parser.add_argument("--numMessage", "-n", type=int, help="number of messages to poll")

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG)

    vae_fb = VAE_FB(model_config, fb_config, logger, FeatureBuilder)
    vae_fb.run()
