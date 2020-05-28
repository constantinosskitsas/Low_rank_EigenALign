
import numpy as np
import argparse
import networkx as nx
import scipy
from networkx.algorithms import bipartite
import time
import os
import sys

from scipy.optimize.optimize import vecnorm
from scipy.sparse import coo_matrix


from numpy.core._multiarray_umath import ndarray
from numpy.linalg import svd
from scipy.linalg import lu
from scipy.linalg._expm_frechet import vec

try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix



















def bipartite_matching(Xsample):
    G = nx.from_numpy_matrix(Xsample)
    D= nx.algorithms.matching.max_weight_matching(G,True)
    return D



def edge_list(param):
    return param


















