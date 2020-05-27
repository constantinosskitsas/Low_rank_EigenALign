import numpy as np
import spicy
from scipy.sparse import coo_matrix

def EigenAlign(A, B, s1, s2, s3, iters):
    if np.shape(A, 0) <= np.shape(B, 0):

        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper(A, B, s1, s2, s3, iters)
        return ei, ej, Xmat, weight, conserved_edges
    else:
        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper(B, A, s1, s2, s3, iters)
        return ej, ei, Xmat.conj(), weight, conserved_edges


def EigenAlign_helper(A, B, s1, s2, s3, iters):
    # error checks
    gam1 = s1 + s2 - 2 * s3
    gam2 = s3 - s2
    gam3 = s2

    nA = np.shape(A, 0)
    nB = np.shape(B, 0)

    AkronB = np.kron(A, B)
    AkronE = np.kron(A, np.ones(nB, nB))
    EkronB = np.kron(np.ones(nA, nA), B)
    EkronE = np.kron(np.ones(nA, nA), np.ones(nB, nB))

    M = gam1 * AkronB + gam2 * AkronE + gam2 * EkronB + gam3 * EkronE

    # Power iteration
    X = np.divide(np.ones(nA @ nB, 1), (nA @ nB))
    X = np.divide(X, np.norm(X, 1))

    x = np.copy(X)
    for i in range(iters):
        y = M @ x
        x = np.divide(y, np.norm(y))
        lam = x.conj() @ y
        print(lam)
    Xmat = x.reshape(nB, nA)

    # for i = 1:iters
    #   X = M*X
    #   X = X./norm(X,2)
    # end
    # X
    # Xmat = reshape(X,nB,nA)

    # Run Hungarian method
    # Xmat = Xmat'
    # ej = munkres(-Xmat)
    # ei = 1:length(ej)
    # ids = find(ej)
    # ej = ej[ids]
    # ei = ei[ids]

    # or bipartite matching
    # try using intmatch
    Xmat = Xmat.conj()
    ei, ej = file1.edge_list(file1.bipartite_matching(spicy.sparse.csc_matrix(Xmat)))

    MATCHING = spicy.sparse.csc_matrix(ei, ej, 1, nA, nB)
    weight = X.conj() @ MATCHING.conj()[:]
    Ai = A[ei, ei]
    Bi = B[ej, ej]
    conserved_edges = np.nnz(Ai * Bi) / 2
    return ei, ej, x, lam  # ,Xmat,weight,conserved_edges