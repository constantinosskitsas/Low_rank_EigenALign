import numpy as np
import scipy
from scipy.optimize.optimize import vecnorm
from numpy import linalg as la
def reversetoPython(ma,mb):
    ma1=np.zeros(ma-1)
    mb1=np.zeros(mb-1)
    for i in range(1,ma):
        ma1[i-1]=ma[i]
    for i in range(1, mb):
        mb1[i - 1] = mb[i]
    mb1=mb1-1
    ma1=ma1-1
    return ma1,mb1
def evaluate_erdosreyni_experiment(A, B, ma, mb):
    nA = np.shape(A)[0]
    print(ma)
    temp=np.ones(np.shape(A)[0],int)
    temp1=np.ones(np.shape(ma)[0],int)
    matest=np.zeros(np.shape(A)[0],int)
    mbtest=np.zeros(np.shape(A)[0],int)
    ma=ma-1
    mb=mb-1
    matemp=range(0,nA)
    mbtemp=range(nA-1, -1, -1)
    for i in range(nA):
        matest[i]=i
    count=0;
    for i in range(nA-1, -1, -1):
        mbtest[count]=i
        count=count+1
    P = scipy.sparse.csc_matrix((temp,(matest,mbtest)),shape=(nA, nA)).toarray()
    Ptil = scipy.sparse.csc_matrix((temp1,(ma, mb)),shape=(nA, nA)).toarray()
    print(la.norm(P - Ptil,ord=1))
    B=(la.norm(P - Ptil, ord=1,axis=0))
    count=0
    for i in range(len(B)):
        count=count+B[i]
    print(count)
    recov = 1-(count / (2 * nA))
    return recov