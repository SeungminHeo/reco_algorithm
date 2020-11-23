# Reco_algorithm

Recommendation algorithm set

## Reco Setting

### Installing Implicit
```
pip install implicit
```



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