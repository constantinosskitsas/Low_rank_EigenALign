import networkx as nx
import numpy as np
from scipy.linalg._expm_frechet import vec
import scipy
scipy.sparse.linalg.norm
from newbound_methods import newbound_rounding_lowrank_evaluation_relaxed


def main(temp1=None):
    nA = 6
    ma = [0, 1, 2, 3, 4, 5]
    mb = [5, 4, 1, 3, 2, 0]
    nA = 6
    temp = np.ones(nA, int)
    matemp=range(0,nA)
    mbtemp=range(nA-1, -1, -1)
    P = scipy.sparse.csc_matrix((temp, (matemp, mbtemp)), shape=(nA, nA))
    Ptil = scipy.sparse.csc_matrix((temp, (ma, mb)), shape=(nA, nA))
    recov = 1 - (scipy.sparse.linalg.norm(P - Ptil, ord=1) / (2 * nA))
    nA = 6
    ma1 = [1, 2, 3, 4, 5, 6]
    mb1 = [6, 5, 2, 4, 3, 1]
    nA1 = 6
    temp1 = np.ones(nA, int)
    matemp1=range(1,nA+1)
    mbtemp1=range(nA, 0, -1)

    P1 = scipy.sparse.csc_matrix((temp1, (matemp1, mbtemp1)), shape=(nA+1, nA+1))
    print((P))
    Ptil1 = scipy.sparse.csc_matrix((temp1, (ma1, mb1)), shape=(nA+1, nA+1))
    print((Ptil))
    print(scipy.sparse.linalg.norm(P-Ptil, ord=1,axis=0))
    print(P-Ptil)
    #recov1 = 1 - (la.norm(P1 - Ptil1, ord=1) / (2 * nA))
    #print(recov1)

if __name__ == "__main__":
	main()