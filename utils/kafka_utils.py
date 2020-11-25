import json
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from itertools import groupby
from operator import itemgetter
from confluent_kafka import Consumer, TopicPartition, Message
from typing import List, Dict


def logging_time(original_fn):
    @wraps(original_fn)
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            "실행시간[{}]: {} Sec".format(
                original_fn.__name__, round(end_time - start_time, 2)
            )
        )
        return result

    return wrapper_fn


class KafkaTopicConsumer:
    def __init__(self, kafka_config: Dict):
        self.config = kafka_config

    @staticmethod
    def json_deserializer(message: Message) -> Dict:
        return json.loads(message.value().decode("UTF-8"))

    def fetch_batch_messages(self, group_id: str, topic_name: str, num_messages: int) -> List[Dict]:
        """
        Fetch messages for lagged messages. Update parameter or models by N messages.
        :param group_id: (String) configure group_id for Kafka consumer
        :param topic_name: (String) configure broker topic for Kafka consumer
        :param num_messages: (String) the number of message to fetch
        :return: List of Json Messages, especially List of Dictionary in Python
        """
        consumer = Consumer(
            {
                "bootstrap.servers": self.config["kafka_config"]["bootstrap_servers"],
                "auto.offset.reset": self.config["kafka_config"]["auto_offset_reset"],
                "group.id": group_id,
            }
        )
        consumer.subscribe([topic_name])
        msgs = consumer.consume(num_messages, timeout=5)

        return [self.json_deserializer(msg) for msg in msgs]

    def fetch_batch_messages_by_time(self, group_id: str, topic_name: str, time_diff_hours: int):
        """
        Fetch messages from selected time interval
        :param group_id: (String) configure group_id for Kafka consumer
        :param topic_name: (String) configure broker topic for Kafka consumer
        :param time_diff_hours: (Integer) The time interval you want to rewind in hours.
        :return: List of Json Messages, especially List of Dictionary in Python
        """
        consumer = Consumer(
            {
                "bootstrap.servers": self.config["kafka_config"]["bootstrap_servers"],
                "auto.offset.reset": self.config["kafka_config"]["auto_offset_reset"],
                "group.id": group_id,
            }
        )

        KST = timezone(timedelta(hours=9))
        date = (datetime.now(tz=KST) - timedelta(hours=time_diff_hours)).timestamp() * 1000
        partition_list = [TopicPartition(topic_name, i, int(date)) for i in range(0, 60)]
        offset_for_times = consumer.offsets_for_times(partition_list)

        consumer.assign(offset_for_times)
        msgs = consumer.consume(time_diff_hours * 5000, timeout=5)

        return [self.json_deserializer(msg) for msg in msgs]


class KafkaFeatureBuilder(KafkaTopicConsumer):
    def __init__(self, kafka_config: Dict):
        super().__init__(kafka_config)

    @logging_time
    def CF(self, time_diff_hours: int) -> str:
        """
        Feature building for Collaborative Filtering.
        :param group_id: (String) configure group_id for Kafka consumer
        :param topic_name: (String) configure broker topic for Kafka consumer
        :param time_diff_hours: (Integer) The time interval you want to rewind in hours.
        :return: (String) Feature for CF.
        """
        feature_message = self.fetch_batch_messages_by_time(
            group_id=self.config["kafka_config"]["consumer_groups"]["cf_model_feed_by_time"],
            topic_name=self.config["kafka_topics"]["click_log"],
            time_diff_hours=time_diff_hours)
        return self.cf_feature_build(self.count_by_piwikId_itemId(feature_message))

    @logging_time
    def GC(self, time_diff_hours: int, topN: int) -> str:
        """
        Feature building for Global Click.
        :param group_id: (String) configure group_id for Kafka consumer
        :param topic_name: (String) configure broker topic for Kafka consumer
        :param time_diff_hours: (Integer) The time interval you want to rewind in hours.
        :param topN: (Integer) The number of topN items to calcuration
        :return: (String) Feature for GC.
        """
        feature_message = self.fetch_batch_messages_by_time(
            group_id=self.config["kafka_config"]["consumer_groups"]["ranking_by_time"],
            topic_name=self.config["kafka_topics"]["click_log"],
            time_diff_hours=time_diff_hours)
        return self.gc_feature_build(self.count_by_itemId(feature_message), topN=topN)

    @logging_time
    def CategoryGC(self, time_diff_hours: int, topN: int) -> str:
        """
        Feature building for Global Click By Category.
        :param group_id: (String) configure group_id for Kafka consumer
        :param topic_name: (String) configure broker topic for Kafka consumer
        :param time_diff_hours: (Integer) The time interval you want to rewind in hours.
        :param topN: (Integer) The number of topN items to calcuration
        :return: (String) Feature for GC.
        """
        feature_message = self.fetch_batch_messages_by_time(
            group_id=self.config["kafka_config"]["consumer_groups"]["category_ranking_by_time"],
            topic_name=self.config["kafka_topics"]["click_log"],
            time_diff_hours=time_diff_hours)
        return self.category_gc_feature_build(self.count_by_categoryId_itemId(feature_message), topN=topN)

    @staticmethod
    @logging_time
    def cf_feature_build(processed_count: List[Dict]) -> str:
        sorted_click_count = sorted(processed_count, key=itemgetter("piwikId", "clickCount"), reverse=True)
        groupby_click_count = groupby(sorted_click_count, key=itemgetter("piwikId"))

        feature_string = "{"
        for enum, kv in enumerate(groupby_click_count):
            if enum > 0:
                feature_string += ", "
            feature_string += '"' + kv[0] + '": {'
            temp_string = ""
            for e, v in enumerate(kv[1]):
                if e > 0:
                    temp_string += ", "
                temp_string += '"' + v["itemId"] + '": ' + str(v["clickCount"])
            feature_string += temp_string + "}"
        feature_string += "}"

        return feature_string

    @staticmethod
    @logging_time
    def gc_feature_build(processed_count: List[Dict], topN: int) -> str:
        sorted_click_count = sorted(processed_count, key=itemgetter("clickCount"), reverse=True)

        feature_string = "{"
        for enum, kv in enumerate(sorted_click_count):
            if enum > topN - 1:
                break
            if enum > 0:
                feature_string += ", "
            feature_string += '"' + kv["itemId"] + '": ' + str(kv["clickCount"])
        feature_string += "}"

        return feature_string

    @staticmethod
    @logging_time
    def category_gc_feature_build(processed_count: List[Dict], topN: int) -> str:
        sorted_click_count = sorted(processed_count, key=itemgetter("categoryId", "clickCount"), reverse=True)
        groupby_click_count = groupby(sorted_click_count, key=itemgetter("categoryId"))

        feature_string = "{"
        for enum, kv in enumerate(groupby_click_count):
            if enum > topN - 1:
                break
            if enum > 0:
                feature_string += ", "
            feature_string += '"' + str(kv[0]) + '": {'
            temp_string = ""
            for e, v in enumerate(kv[1]):
                if e > 0:
                    temp_string += ", "
                temp_string += '"' + v["itemId"] + '": ' + str(v["clickCount"])
            feature_string += temp_string + "}"
        feature_string += "}"

        return feature_string

    @staticmethod
    @logging_time
    def count_by_categoryId_itemId(messages: List[Dict]) -> List[Dict]:
        messages = filter(lambda x: x["categoryId"] is not None, messages)
        sorted_msgs = sorted(messages, key=itemgetter('categoryId', 'itemId'))
        groupby_msgs = groupby(sorted_msgs, key=itemgetter('categoryId', 'itemId'))

        processed_msgs = [{
            "categoryId": key[0],
            "itemId": key[1],
            "clickCount": list(value).__len__(),
        } for key, value in groupby_msgs]

        return processed_msgs

    @staticmethod
    @logging_time
    def count_by_piwikId_itemId(messages: List[Dict]) -> List[Dict]:
        sorted_msgs = sorted(messages, key=itemgetter('piwikId', 'itemId'), reverse=True)
        groupby_msgs = groupby(sorted_msgs, key=itemgetter('piwikId', 'itemId'))

        processed_msgs = [{
            "piwikId": key[0],
            "itemId": key[1],
            "clickCount": list(value).__len__()
        } for key, value in groupby_msgs]

        return processed_msgs

    @staticmethod
    @logging_time
    def count_by_itemId(messages: List[Dict]) -> List[Dict]:
        sorted_msgs = sorted(messages, key=itemgetter('itemId'), reverse=True)
        groupby_msgs = groupby(sorted_msgs, key=itemgetter('itemId'))

        processed_msgs = [{
            "itemId": key,
            "clickCount": list(value).__len__()
        } for key, value in groupby_msgs]

        return processed_msgs
