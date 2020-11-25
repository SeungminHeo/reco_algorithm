from argparse import ArgumentParser

from utils.kafka_config import CONFIG
from utils.kafka_utils import KafkaFeatureBuilder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runningEnvironment", "-re", type=str, help="environment that runs reco engine.",
                        choices=["server", "local"])

    args = parser.parse_args()
    FeatureBuilder = KafkaFeatureBuilder(CONFIG[args.runningEnvironment])

    ## feature build by time
    print(FeatureBuilder.CF(
        time_diff_hours=3
    ))

    print("\n" * 5)
    ## Global Click Ranking by time
    print(FeatureBuilder.GC(
        time_diff_hours=3,
        topN=100
    ))

    print("\n" * 5)
    ## Global CategoryClick Ranking by time
    print(FeatureBuilder.CategoryGC(
        time_diff_hours=3,
        topN=100
    ))
