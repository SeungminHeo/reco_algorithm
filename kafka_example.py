from argparse import ArgumentParser

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--numMessage", "-n", type=int, help="number of messages to poll")

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG)

    ## feature build by time
    print(FeatureBuilder.CF(
        group_id=CONFIG["kafka_config"]["consumer_groups"]["cf_model_feed_by_time"],
        topic_name=CONFIG["kafka_topics"]["click_log"],
        time_diff_hours=3
    ))

    print("\n" * 5)
    ## Global Click Ranking by time
    print(FeatureBuilder.GC(
        group_id=CONFIG["kafka_config"]["consumer_groups"]["ranking_by_time"],
        topic_name=CONFIG["kafka_topics"]["click_log"],
        time_diff_hours=3,
        topN=100
    ))

    print("\n" * 5)
    ## Global CategoryClick Ranking by time
    print(FeatureBuilder.CategoryGC(
        group_id=CONFIG["kafka_config"]["consumer_groups"]["category_ranking_by_time"],
        topic_name=CONFIG["kafka_topics"]["click_log"],
        time_diff_hours=3,
        topN=100
    ))
