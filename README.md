# Reco_algorithm

Recommendation algorithm set


## Kafka Setting 

### Installing Confluent-kafka 
```
pip install confluent-kafka
```

### Installing librdkafka

Confluent Kafka는 C extension 기반으로 작성되어있으므로 관련 설치가 필요함.

참고 : `https://github.com/edenhill/librdkafka`

You can download and install librdkafka using the vcpkg dependency manager:

```
#### Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install

#### Install librdkafka
./vcpkg install librdkafka
```

### Kafka Example

```
python3 kafka_example.py -n 1000
```

### MongoDB Example

```
client = mongo_utils.MongoConnection('als')

# 하나의 유저에 대한 결과만 작성할 때
client.write_one({'piwikId': 'userId1', 'recoResult': {'itemId1': 10, 'itemId2': 9}})

# 여러 유저에 대한 결과를 작성할 때
client.write_many([{'piwikId': 'userId1', 'recoResult': {'itemId1': 10, 'itemId2': 9}}, {'piwikId': 'userId2', 'recoResult': {'itemId1': 6, 'itemId2': 10}}])

# 하나의 유저에 대한 결과만 가져올 때
client.load_one('userId1')

# 여러 유저에 대한 결과를 가져올 때
client.load_many(['userId1', 'userId2'])

# 모든 결과를 가져올 때
client.load_all()
```
