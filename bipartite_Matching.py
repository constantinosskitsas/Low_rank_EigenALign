
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


def bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n):
    print("primal1")
    ai = ai/np.amax(abs(ai))
    alpha = np.zeros((m), float)
    bt = np.zeros((m + n), float)  # beta
    queue = np.zeros((m), int)
    t = np.zeros((m + n), int)
    match1 = np.zeros((m), int)
    match2 = np.zeros(( m + n), int)
    tmod = np.zeros(( m + n), int)
    ntmod = 0
    m=m-1
    n=n-1
    # initialize the primal and dual variables
    for i in range(m):
        for rpi in range(rp[i], rp[i + 1]):
            if ai[rpi] > alpha[i]:
                alpha[i] = ai[rpi]
# dual variables (bt) are initialized to 0 already
# match1 and match2 are both 0, which indicates no matches

    i = 0
    while i <= m:
        print("primal2")
        print(i)
        for j in range(ntmod):
            t[tmod[j]] = 0

        ntmod = 0
# add i to the stack
        head = 0
        tail = 0
        queue[head] = i
        while head <= tail and match1[i] == 0:
            print("primal3")
            k = queue[head]
            for rpi in range(rp[k], rp[k+1]):
                print("primal4")
                j = ci[rpi]
                if ai[rpi] < alpha[k] + bt[j] - 1e-7:
                   continue
                elif t[j] == 0:
                    tail = tail + 1
                    if tail <= m:
                        queue[tail] = match2[j]

                    t[j] = k
                    ntmod = ntmod + 1
                    tmod[ntmod] = j
                    if match2[j] < 1:
                        while j > 0:
                            print("primal5")
                            match2[j] = t[j]
                            k = t[j]
                            temp = match1[k]
                            match1[k] = j
                            j = temp
                        break
            head = head + 1
        print("hi")
        if match1[i] < 1:
            theta = np.math.inf
            for j in range(head-1):
                print("primal6")
                t1 = queue[j]
                for rpi in range(rp[t1], rp[t1+1]):
                    print("primal7")
                    t2 = ci[rpi]
                    if t[t2] == 0 and alpha[t1] + bt[t2] - ai[rpi] < theta:
                        theta = alpha[t1] + bt[t2] - ai[rpi]
            print("primal8")
            for j in range(head - 1):
                alpha[queue[j]] -= theta

            for j in range(ntmod):
                bt[tmod[j]] += theta
            #continue

        i = i + 1
    print("primal half")
    val = 0
    for i in range(m):
        for rpi in range(rp[i], rp[i + 1]):
            if ci[rpi] == match1[i]:
                val = val + ai[rpi]

    noute = 0
    for i in range(m):
        if match1[i] <= n:
            noute = noute + 1
    return match1 #for now
   # return m,n,val,noute,match1





def bipartite_matching_setup(nzi,nzj,nzv):
    print("setup")
    nedges = len(nzi)
    m=len(nzi)
    n=len(nzj)

    rp = np.ones(( m + 1),int) # csr matrix with extra edges
    ci = np.zeros(( nedges + m),int)
    ai = np.zeros(( nedges + m),int)

    rp[0] = 0
    for i in range(nedges):
        rp[nzi[i] + 1] = rp[nzi[i] + 1] + 1

    rp = np.cumsum(rp)

    for i in range(nedges):
        ai[rp[nzi[i]] + 1] = nzv[i]
        ci[rp[nzi[i]] + 1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]] + 1


    for i in range(m-1): # add the extra edges
        ai[rp[i] + 1] = 0
        ci[rp[i] + 1] = n + i
        rp[i] = rp[i] + 1

    print("setup half")
    # restore the row pointer array
    for i in range(m - 1, -1, -1):
        rp[i + 1] = rp[i]

    rp[0] = 0
    rp = rp+ 1

    # check for duplicates in the data
    colind = np.zeros(( m + n),int)
    for i in range(m):
        for rpi in range(rp[i],rp[i+1]):
            if colind[ci[rpi]] == 1:
                print("error -bipartite_matching:duplicateEdge")

            colind[ci[rpi]] = 1


        for rpi in range(rp[i],rp[i+1]):
            colind[ci[rpi]] = 0

    bipartite_matching_primal_dual(rp,ci,ai,[],m,n)



def bipartite_matching_setup1(nzi,nzj,nzv):

    nedges = len(nzi)-1
    m = len(nzi)-1
    n = len(nzj)-1
    rp = np.ones((m + 1+1), int)  # csr matrix with extra edges
    ci = np.zeros((nedges + m+2), int)
    ai = np.zeros((nedges + m+2), int)
    tripi = np.zeros( (nedges + m+2),int)
# 1. build csr representation with a set of extra edges from vertex i to
# vertex m+i

    rp[0] = 0
    for i in range(nedges):
        rp[nzi[i] + 1] = rp[nzi[i] + 1] + 1


    rp = np.cumsum(rp)
    for i in range(nedges):
        tripi[rp[nzi[i]]+1]=i
        ai[rp[nzi[i]] + 1] = nzv[i]
        ci[rp[nzi[i]] + 1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]] + 1

    for i in range(m): # add the extra edges
        tripi[rp[i] + 1] = -1
        ai[rp[i] + 1] = 0
        ci[rp[i] + 1] = n + i
        rp[i] = rp[i] + 1

    for i in range(m - 1, -1, -1):
        rp[i + 1] = rp[i]

    rp[0] = 0
    rp = rp+ 1
# restore the row pointer array
    colind = np.zeros((m + n), int)
# check for duplicates in the data
    for i in range(m):
        for rpi in range(rp[i],rp[i+1]):
            if colind[ci[rpi]] == 1:
                print("error -bipartite_matching:duplicateEdge")

            colind[ci[rpi]] = 1


        for rpi in range(rp[i],rp[i+1]):
            colind[ci[rpi]] = 0

    bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n)

def bipartite_matching1(nzi,nzj,nzv):
    return bipartite_matching_setup1(nzi,nzj,nzv)



def bipartite_matching(Xsample):
    G = nx.from_numpy_matrix(Xsample)
    D= nx.algorithms.matching.max_weight_matching(G,True)
    return D



def edge_list(param):
    return param


















