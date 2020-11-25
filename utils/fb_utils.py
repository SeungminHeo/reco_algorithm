from typing import List
import json
import pandas as pd
import numpy as np
from scipy import sparse

def to_sparse_tuples(data:json, user_n=0, item_n=0,
                     user2idx:dict=None, item2idx:dict=None)\
                     -> (List[tuple], dict, dict):
	if user_n > 0:
        f = lambda x: len(x[1]) > user_n
        data = dict(list(filter(f, data.items())))
    if user2idx is None:
		user2idx = {j:i for i,j in enumerate(data.keys())}
	if item2idx is None:
        items = set()
        for i in list(data.values()):
            items = items.union(set(i))
        item2idx = {j:i for i,j in enumerate(items)}
        sparse_tuples = []
    for i in data:
        items = data[i]
        for j in items:
            sparse_tuples.append((user2idx[i], item2idx[j], items[j]))
    return sparse_tuples, user2idx, item2idx

def to_sparse(data:List[tuple], relevance:List[int]=None) -> sparse.csr_matrix:
    data = np.array(data)
    if relevance is None:
        relevance = np.ones_like(data[:,0])
    else:
        relevance = np.array(val)
    return sparse.csr_matrix((val, (data[:,0], data[:,1])),
							 dtype="float32",
							 shape=(data[:,0].max()+1, data[:,1].max()+1))
