import json
from datetime import datetime, timedelta, timezone
from itertools import groupby
from operator import itemgetter
from confluent_kafka import Consumer, TopicPartition, Message
from typing import List, Dict


class KafkaTopicConsumer:
    def __init__(self, kafka_config: dict):
        self.config = kafka_config

    def fetch_batch_messages(self, group_id: str, topic_name: str, num_messages: int) -> List[Dict]:
        """
        Fetch messages for lagged messages. Update parameter or models by N messages.
        :param group_id: (String) consumer group name
        :param topic_name: (String) topic to fetch
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
        msgs = consumer.consume(num_messages, timeout=10)

        return [json_deserializer(msg) for msg in msgs]

    def fetch_batch_messages_by_time(self, group_id: str, topic_name: str, time_diff_hours: int):
        """
        Fetch messages from selected time interval
        :param group_id: (String) consumer group name
        :param topic_name: (String) topic to fetch
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
        date = (datetime.now(tz=KST) - timedelta(hours=time_diff_hours)).timestamp()*1000
        partition_list = [TopicPartition(topic_name, i, int(date)) for i in range(0, 60)]
        offset_for_times = consumer.offsets_for_times(partition_list)

        consumer.assign(offset_for_times)
        msgs = consumer.consume(time_diff_hours*5000, timeout=20)

        return [json_deserializer(msg) for msg in msgs]


def json_deserializer(message: Message) -> Dict:
    return json.loads(message.value().decode("UTF-8"))


def CF_feature_build(processed_count: List[Dict]) -> str:
    sorted_click_count = sorted(processed_count, key=itemgetter("piwikId", "clickCount"), reverse=True)
    groupby_click_count = groupby(sorted_click_count, key= itemgetter("piwikId"))

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


def Ranking_feature_build(processed_count: List[Dict], topN: int) -> str:
    sorted_click_count = sorted(processed_count, key=itemgetter("clickCount"), reverse=True)

    feature_string = "{"
    for enum, kv in enumerate(sorted_click_count):
        if enum > topN-1:
            break
        if enum > 0:
            feature_string += ", "
        feature_string += '"' + kv["itemId"] + '": ' + str(kv["clickCount"])
    feature_string += "}"

    return feature_string


def Category_Ranking_feature_build(processed_count: List[Dict], topN: int) -> str:
    sorted_click_count = sorted(processed_count, key=itemgetter("categoryId", "clickCount"), reverse=True)
    groupby_click_count = groupby(sorted_click_count, key=itemgetter("categoryId"))

    feature_string = "{"
    for enum, kv in enumerate(groupby_click_count):
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


def count_by_categoryId_itemId(messages: List[Dict]) -> List[Dict]:
    messages = filter(lambda x: x["categoryId"] != None, messages)
    sorted_msgs = sorted(messages, key=itemgetter('categoryId', 'itemId'))
    groupby_msgs = groupby(sorted_msgs, key=itemgetter('categoryId', 'itemId'))

    processed_msgs = [{
        "categoryId": key[0],
        "itemId": key[1],
        "clickCount": list(value).__len__(),
    } for key, value in groupby_msgs]

    return processed_msgs



def count_by_piwikId_itemId(messages: List[Dict]) -> List[Dict]:
    sorted_msgs = sorted(messages, key=itemgetter('piwikId', 'itemId'), reverse=True)
    groupby_msgs = groupby(sorted_msgs, key=itemgetter('piwikId', 'itemId'))

    processed_msgs = [{
        "piwikId": key[0],
        "itemId": key[1],
        "clickCount": list(value).__len__()
    } for key, value in groupby_msgs]

    return processed_msgs


def count_by_itemId(messages: List[Dict]) -> List[Dict]:
    sorted_msgs = sorted(messages, key=itemgetter('itemId'), reverse=True)
    groupby_msgs = groupby(sorted_msgs, key=itemgetter('itemId'))

    processed_msgs = [{
        "itemId": key,
        "clickCount": list(value).__len__()
    } for key, value in groupby_msgs]

    print(processed_msgs.__len__())

    return processed_msgs






