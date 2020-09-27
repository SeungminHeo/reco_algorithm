import sys
import unittest
import yaml
import logging
import logging.config
sys.path.append("../model")
from rankfusion import Rankfusion


class Test(unittest.TestCase):
    def setUp(self):
        configs = yaml.load(open("../conf/config.yml").read(), Loader=yaml.Loader)
        model_config = configs["model_config"]['rankfusion']
        logging_config = configs["logging_config"]

        # insert "logging_config" values into logging config
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger('test')
        self.rank = Rankfusion(model_config, logger)

    def test_case1(self):
        als = ["0", "1", "2", "3"]
        cb = ["2", "1", "3", "0"]
        cf = ["0", "1", "4", "2"]
        algos = [als, cb, cf]
        self.rank.make_rank(algos)


if __name__ == '__main__':
    unittest.main()
