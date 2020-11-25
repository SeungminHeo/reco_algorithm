import yaml
import json
import logging.config
from datetime import datetime

from argparse import ArgumentParser

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder
from utils.mongo_connect import MongoConnection
from model.rankfusion import Rankfusion


class Reco:
    def __init__(self, reco_config, logger, featurebuilder, als_client, reco_client):
        # logger
        self.logger = logger
        self.logger.info('start Reco')

        # kafka
        self.FeatureBuilder = featurebuilder

        # reco config
        self.reco_config = reco_config

        # Rank fusion
        model_config = configs["model_config"]['rankfusion']
        self.rank = Rankfusion(model_config, logger)

        # mongo
        self.als_client = als_client
        self.reco_client = reco_client

    def reco_result(self, data):
        return {x['piwikId']: list(x['recoResult'].keys()) for x in data}

    def load_model(self, time_diff_hours: int):
        als_mongo = als_client.load_all()
        gc_kafka = json.loads(self.FeatureBuilder.GC(
            time_diff_hours=time_diff_hours,
            topN=100
        ))
        cate_gc_kafka = json.loads(FeatureBuilder.CategoryGC(
            time_diff_hours=time_diff_hours,
            topN=100
        ))
        self.als_reco = self.reco_result(als_mongo)
        self.gc_reco = list(gc_kafka.keys())[:20]

    def reco(self):
        final_reco = []
        for user_id, reco_items in self.als_reco.items():
            user_reco = {}
            if self.reco_config['gc'].get('is_used'):
                reco_items = self.rank.make_rank([reco_items, self.gc_reco])
            else:
                reco_items = self.rank.make_rank([reco_items])

            user_reco["piwikId"] = user_id
            user_reco['recoResult'] = reco_items
            final_reco.append(user_reco)
        reco_client.write_many(final_reco)

    def run(self, time_diff_hours):
        while True:
            start = datetime.utcnow()
            self.load_model(time_diff_hours=time_diff_hours)
            self.reco()
            end = datetime.utcnow()
            logger.info("Training model for recommendation is done."
                        "Training time is %s sec." % (end - start) )



if __name__ == "__main__":
    configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)
    reco_config = configs['main_process']['reco']
    logging_config = configs["logging_config"]

    # insert "logging_config" values into logging config
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('ALS_FeatureBuilder')

    # kafka parser
    parser = ArgumentParser()
    parser.add_argument("--hours", type=int, help="interval in hours from now to get train data")
    parser.add_argument("--runningEnvironment", "-re", type=str, help="environment that runs reco engine.", choices=["server", "local"] )

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG[args.runningEnvironment])

    als_client = MongoConnection('als', args.runningEnvironment)
    reco_client = MongoConnection('recoResult', args.runningEnvironment)

    reco = Reco(reco_config, logger, FeatureBuilder, als_client, reco_client)
    reco.run(args.hours)
