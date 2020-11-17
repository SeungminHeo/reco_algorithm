from pathlib import Path
import yaml

# environment = os.environ['ENVIRONMENT']
with Path(f"conf/kafka_config.yaml").open() as config_file:
    CONFIG = yaml.load(config_file, Loader=yaml.FullLoader)