# EigenAlign Algorithm
# input: two matrices A and B
# input: s1,s2,s3
import numpy as np
import spicy as spicy
import scipy
from scipy.sparse import coo_matrix
#TODO: input types
#TODO: error checks
from file1 import bipartite_matching, edge_list


def EigenAlign_matching(A,B,s1,s2,s3,iters):
  if np.shape(A,0)<=np.shape(B,0):

    ei,ej = EigenAlign_helper_matching(A,B,s1,s2,s3,iters)
    return ei,ej
  else:
    ei,ej = EigenAlign_helper_matching(B,A,s1,s2,s3,iters)
    return ej,ei

def EigenAlign_helper_matching(A,B,s1,s2,s3,iters):

  gam1 = s1+s2-2*s3
  gam2 = s3-s2
  gam3 = s2

  nA = np.shape(A,0)
  nB = np.shape(B,0)

  # AkronB = kron(A,B)
  # AkronE = kron(A,ones(nB,nB))
  # EkronB = kron(ones(nA,nA),B)
  # EkronE = kron(ones(nA,nA),ones(nB,nB))
  #
  # M = gam1*AkronB + gam2*AkronE + gam2*EkronB + gam3*EkronE

  Eb = np.ones(int,nB,nB)
  Ea = np.ones(int,nA,nA)

  # Power iteration
  # X = ones(nA*nB,1)./(nA*nB)
  # X = X./norm(X,1)

  X = np.ones(nB,nA)
  X = np.divide(X,sum(X))

  for i in range(iters):
    X = gam1@B@X@A.conj() + gam2@Eb@X@A.conj() + gam2@B@X@Ea + gam3@Eb@X@Ea
    X = np.divide(X,sum(X))
  Xmat = X.mat.reshape(X,nB,nA)

  # Run Hungarian method
  # Xmat = Xmat'
  # ej = munkres(-Xmat)
  # ei = 1:length(ej)
  # ids = find(ej)
  # ej = ej[ids]
  # ei = ei[ids]

  # or bipartite matching
  ej,ei = edge_list(bipartite_matching(spicy.sparse.csc_matrix(Xmat)))
  ## bmatching start
  P = spicy.sparse.csc_matrix(ei,ej,0,nA,nB)
  ej.pop(0)
  P = P + spicy.sparse.csc_matrix(ei[1:-2],ej,0,nA,nB)
  Xsample = P.conj()*Xmat
  ej,ei = edge_list(bipartite_matching(spicy.sparse.csc_matrix(Xsample)))
  ## bmatching end
  return ei,ej

