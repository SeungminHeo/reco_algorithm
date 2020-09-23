import logging
import logging.config

import yaml
from easydict import EasyDict

import numpy as np
logging.config.fileConfig('../conf/logging.conf')


class Rankfusion:
    """Python implementation for RankFusion.

    Implementation of Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.

    Reference: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.150.2291&rep=rep1&type=pdf"""
    def __init__(self):
        params = yaml.load(open("../conf/model.yml").read(), Loader=yaml.Loader)['model']['rankfusion']['params']
        self.k = params.get('k')

        self.logger = logging.getLogger('Rankfusion')
        self.logger.info('start Rankfusion ')
        self.logger.info('Rankfusion parameters -> (k : %d)' % self.k)

    def make_rank(self, algos):
        '''
        algos : a list of rankings ex) [[als..], [w2v...], [cb..]]
        return recomendation rank
        '''
        pools = set()
        for algo_pool in algos:
            pools.update(algo_pool)

        rets = {}  # final recomnedation
        for item in pools:
            for algo_pool in algos:
                if item in algo_pool:
                    rets[item] = rets.setdefault(item, 0) + 1.0/(self.k + algo_pool.index(item)+1)
        rets = {k: v for k, v in sorted(rets.items(), key=lambda item: item[1])}
        self.logger.info("Recomendation items : %s" % rets)
        return rets
