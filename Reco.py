import sys
import yaml
import json
import logging.config

from argparse import ArgumentParser

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder
from utils.mongo_connect import MongoConnection

sys.path.append("./model")
from rankfusion import Rankfusion

class Reco:
    def __init__(self, reco_config, logger, FeatureBuilder, als_client, reco_client):
        ## logger
        self.logger = logger
        self.logger.info('start Reco')

        ## kafka
        self.FeatureBuilder = FeatureBuilder

        ## reco config
        self.reco_config = reco_config

        ## Rank fusion
        model_config = configs["model_config"]['rankfusion']
        self.rank = Rankfusion(model_config, logger)

        ## mongo
        self.als_client = als_client
        self.reco_client = reco_client

    def reco_result(self, data):
        return {x['piwikId']: list(x['recoResult'].keys()) for x in data}

    def load_model(self):
        als_mongo = als_client.load_all()
        gc_kafka = json.loads(self.FeatureBuilder.GC(
            group_id=CONFIG["kafka_config"]["consumer_groups"]["ranking_by_time"],
            topic_name=CONFIG["kafka_topics"]["click_log"],
            time_diff_hours=3,
            topN=100
        ))
        cate_gc_kafka = json.loads(FeatureBuilder.CategoryGC(
            group_id=CONFIG["kafka_config"]["consumer_groups"]["category_ranking_by_time"],
            topic_name=CONFIG["kafka_topics"]["click_log"],
            time_diff_hours=3,
            topN=100
        ))
        self.als_reco = self.reco_result(als_mongo)
        self.gc_reco = list(gc_kafka.keys())[:20]


    def reco(self):
        final_reco = []
        for user_id, reco_items in self.als_reco.items():
            user_reco = {}
            if self.reco_config['gc'].get('is_used'):
                reco_items = list(self.rank.make_rank([reco_items, self.gc_reco]).keys())

            user_reco["piwikId"] = user_id
            user_reco['recoResult'] = reco_items
            final_reco.append(user_reco)

        reco_client.write_many(final_reco)

    def run(self):
        while True:
            self.load_model()
            self.reco()


if __name__ == "__main__":
    configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)
    reco_config = configs['main_process']['reco']
    logging_config = configs["logging_config"]

    # insert "logging_config" values into logging config
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('ALS feature_builder')

    # kafka parser
    parser = ArgumentParser()
    parser.add_argument("--numMessage", "-n", type=int, help="number of messages to poll")

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG)

    als_client = MongoConnection('als')
    reco_client = MongoConnection('recoResult')

    reco = Reco(reco_config, logger, FeatureBuilder, als_client, reco_client)
    reco.run()