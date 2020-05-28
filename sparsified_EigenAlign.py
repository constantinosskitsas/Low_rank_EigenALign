import numpy as np
import scipy
from scipy.sparse import coo_matrix

import bipartite_Matching
from decomposeX import decomposeX


def keep_top_k(v, topk):
    vsortedperm = np.argsort(v, 'mergesort')
    tokeep_ids = vsortedperm[1:topk]
    tokeep_vals = v[tokeep_ids]
    n = len(v)
    w = scipy.sparse.random(n, 1, 0.0)
    w[tokeep_ids] = tokeep_vals
    return w


def sparsified_EigenAlign(A, B, c1, c2, c3, iters, topk):
    Uk, Vk, Wk, W1, W2 = decomposeX(A, B, iters, c1, c2, c3)
    U1 = Uk
    V1 = Vk @ Wk.conj()

    k = np.shape(U1, 1)
    n = np.shape(U1, 0)
    X = coo_matrix(n, n)

    for rowid in range(n):
        u = U1[rowid, :]
        tempvec = np.zeros(n)
        for i in range(len(u)):
            tempvec += u[i] * V1[:, i]
        w = keep_top_k(tempvec, topk)
        X[rowid, np.find(w)] = w[np.find(w)]
    ma, mb = bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching(X))

    return ma, mb, X