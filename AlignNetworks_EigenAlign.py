
import numpy as np
import scipy
from numpy.linalg import svd
from scipy.linalg import lu
from scipy.linalg._expm_frechet import vec

import decomposeX
import bipartite_Matching
import newbound_methods
from greed_match import greedy_match
from greedy_lowrank_EigenAlign import greedy_lowrank


def align_networks_eigenalign(A, B, iters, method, bmatch, default_params=True):
    D = 0
    s1, s2, s3 = find_parameters(A, B)
    if not default_params:
        s1 += 100
        s2 += 10
        s3 += 5
    c1 = s1 + s2 - 2 * s3
    c2 = s3 - s2
    c3 = s2
    Uk, Vk, Wk, W1, W2 = decomposeX.decomposeX_balance_allfactors(A, B, iters + 1, c1, c2, c3)
    Un, Vn = split_balanced_decomposition(Uk, Wk, Vk)
    timematching = 0
    nA = len(A[0])
    nB = len(B[0])

    if method == "lowrank_svd_union":
        U, S, V = np.linalg.svd(Wk)
        U1,S1,V1=svd(Wk)
        U1 = np.dot(np.dot(Uk,U), np.diag(np.sqrt(S)))
        V1 = np.dot(np.dot(Vk,V ), np.diag(np.sqrt(S)))
        X = newbound_methods.newbound_rounding_lowrank_evaluation_relaxed(U1, V1, bmatch) * (10 ** 8)  # bmatch
        avgdeg = map(lambda x: sum(X[x, :] != 0), np.arange(0, np.shape(X)[0], 1))
        avgdeg1 = map(lambda x: sum(X[x, :] != 0), range(np.shape(X)[0]))
        avgdeg = np.array(list(avgdeg)) #np.fromiter(avgdeg, dtype=np.float)
        avgdeg = np.mean(avgdeg)
        Matching = bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching(X))  # 1
        D = avgdeg;  # nnz(X)/prod(size(X))
        #print(list(Matching))

    else:
        print(
            "method should be one of the following: (1)eigenalign,(2)lowrank_unbalanced_best, (3)lowrank_unbalanced_union,(4)lowrank_balanced_best, (5)lowrank_balanced_union,(6)lowrank_Wkdecomposed_best, (7)lowrank_Wkdecomposed_union")
    return Matching, D, timematching


def find_parameters(A, B):
    nB = len(B[0])
    nA = len(A[0])
    nmatches = np.sum(A) * np.sum(B)
    nmismatches = np.sum(A) * ((nB ** 2) - np.sum(B)) + np.sum(B) * ((nA ** 2) - np.sum(A))
    mygamma = nmatches / nmismatches
    myalpha = (1 / mygamma) + 1
    myeps = 0.001
    s1 = myalpha + myeps
    s2 = 1 + myeps
    s3 = myeps
    return s1, s2, s3

def split_balanced_decomposition(Uk, Wk, Vk):
    P, L, U = scipy.linalg.lu(Wk, False)
    Ud = np.diag(np.sqrt(abs(np.diag(U))))
    L2 = np.dot(L, Ud)
    Utemp=np.sqrt(np.diag(U))
    Utemp[np.isnan(Utemp)]=0
    where_are_NaNs = np.isnan(Utemp)
    #for i in range(len(where_are_NaNs)):
      #  if where_are_NaNs[i]==True:
         #   Utemp[where_are_NaNs] = 0

    Utemp2=np.divide(1, Utemp)
    Utemp2[np.isinf(Utemp2)] = 0
    U2 = np.dot(np.diag(Utemp2), U)
    Un = np.dot(Uk,L2)
    Vn = np.dot(Vk,U2.transpose())


    return Un, Vn

def split_svd(Uk, Wk, Vk):
    U, S, V = svd(Wk)
    D = np.diag(np.sqrt(S))
    Unew = Uk @ U @ D
    Vnew = Vk @ V @ D
    return Unew, Vnew

