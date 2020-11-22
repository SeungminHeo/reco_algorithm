from argparse import ArgumentParser

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaTopicConsumer, CF_feature_build, count_by_piwikId_itemId, Ranking_feature_build, \
    count_by_itemId

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--numMessage", "-n", type=int, help="number of messages to poll")

    args = parser.parse_args()
    consumer = KafkaTopicConsumer(CONFIG)

    ## feature build by the number of messages
    print(
        CF_feature_build(
            count_by_piwikId_itemId(
                consumer._fetch_batch_messages(
                    group_id=CONFIG["kafka_config"]["consumer_groups"]["cf_model_feed"],
                    topic_name=CONFIG["kafka_topics"]["click_log"],
                    num_messages=args.numMessage
                )
            )
        )
    )

    print("\n"*20)

    ## feature build by time
    print(
        CF_feature_build(
            count_by_piwikId_itemId(
                consumer._fetch_batch_messages_by_time(
                    group_id=CONFIG["kafka_config"]["consumer_groups"]["cf_model_feed_by_time"],
                    topic_name=CONFIG["kafka_topics"]["click_log"],
                    time_diff_hours=3
                )
            )
        )
    )

    print("\n" * 20)

    ## Global Click Ranking by time
    print(
        Ranking_feature_build(
            count_by_itemId(
                consumer._fetch_batch_messages_by_time(
                    group_id=CONFIG["kafka_config"]["consumer_groups"]["ranking_by_time"],
                    topic_name=CONFIG["kafka_topics"]["click_log"],
                    time_diff_hours=3
                )
            ),
            topN=100
        )
    )

