import logging
import logging.config
import yaml
from easydict import EasyDict
from typing import List
import numpy as np


class Rankfusion:
    """Python implementation for RankFusion.

    Implementation of Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.

    Reference: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.150.2291&rep=rep1&type=pdf"""
    def __init__(self, model_config, logger):
        '''
        model_config : config.yml['model_config']['rankfusion']
        logger : getLogger('Rankfusion')
        '''
        params = model_config['params']
        self.k = params.get('k')

        self.logger = logger
        self.logger.info('start Rankfusion ')
        self.logger.debug('Rankfusion parameters -> (k : %d)' % self.k)

    def make_rank(self, algo_pools: List[str]) -> dict:
        '''
        algo_pools : a list of rankings ex) [[als..], [w2v...], [cb..]]
        return recomendation rank
        '''
        pools = set()
        for algo_pool in algo_pools:
            pools.update(algo_pool)

        Reco_ranking = {}  # final recomnedation
        for item in pools:
            for algo_pool in algo_pools:
                if item in algo_pool:
                    Reco_ranking[item] = Reco_ranking.setdefault(item, 0) + 1.0/(self.k + algo_pool.index(item)+1)
        Reco_ranking = {k: v for k, v in sorted(Reco_ranking.items(), key=lambda item: item[1])}
        self.logger.debug("Recomendation items : %s" % Reco_ranking)
        return Reco_ranking