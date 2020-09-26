import unittest
import sys
import numpy as np
import yaml
import logging
import logging.config
sys.path.append("../model")
from als import ALS


class Test(unittest.TestCase):
    def setUp(self):
        configs = yaml.load(open("../conf/config.yml").read(), Loader=yaml.Loader)
        model_config = configs["model_config"]['als']
        logging_config = configs["logging_config"]

        # insert "logging_config" values into logging config
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger('test')
        self.als = ALS(model_config, logger)

    def test_case1(self):
        r = np.array([[0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
                      [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
                      [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                      [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
                      [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                      [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0]])
        self.als.set_data(r)
        predict = self.als.train()

    def test_case2(self):
        r = np.array([[0, 0, 1],
                      [0, 0, 0],
                      [0, 3, 4],
                      [5, 8, 9]])
        self.als.set_data(r)
        predict = self.als.train()
        

if __name__ == '__main__':
    unittest.main()
