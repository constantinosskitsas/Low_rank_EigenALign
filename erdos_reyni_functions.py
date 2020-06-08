import numpy as np
import scipy
from scipy.optimize.optimize import vecnorm
from numpy import linalg as la

def evaluate_erdosreyni_experiment(A, B, ma, mb):
    nA = np.shape(A)[0]
    temp=np.ones(np.shape(A)[0],int)
    temp1=np.ones(np.shape(ma)[0],int)
    print(nA)
    print(len(ma))
    matemp=range(0,nA)
    mbtemp=range(nA-1, -1, -1)
    P = scipy.sparse.csc_matrix((temp,(matemp,mbtemp)),shape=(nA, nA)).toarray()
    Ptil = scipy.sparse.csc_matrix((temp1,(ma, mb)),shape=(nA, nA)).toarray()
    recov = (la.norm(P - Ptil,ord=1) / (2 * nA))
    return recov