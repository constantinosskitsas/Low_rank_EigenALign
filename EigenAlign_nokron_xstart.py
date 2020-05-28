
import numpy as np
import scipy
import bipartite_Matching


def EigenAlign_nokron_xstart(A, B, s1, s2, s3, iters, Xstart):
    if np.shape(A, 0) <= np.shape(B, 0):

        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper_nokron_xstart(A, B, s1, s2, s3, iters, Xstart)
        return ei, ej, Xmat, weight, conserved_edges
    else:
        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper_nokron_xstart(B, A, s1, s2, s3, iters, Xstart)
        return ej, ei, Xmat.conj(), weight, conserved_edges

def EigenAlign_helper_nokron_xstart(A, B, s1, s2, s3, iters, Xstart):
    # error checks
    gam1 = s1 + s2 - 2 * s3
    gam2 = s3 - s2
    gam3 = s2

    nA = np.shape(A, 0)
    nB = np.shape(B, 0)

    # AkronB = kron(A,B)
    # AkronE = kron(A,ones(nB,nB))
    # EkronB = kron(ones(nA,nA),B)
    # EkronE = kron(ones(nA,nA),ones(nB,nB))
    #
    # M = gam1*AkronB + gam2*AkronE + gam2*EkronB + gam3*EkronE

    Eb = np.ones(int, nB, nB)
    Ea = np.ones(int, nA, nA)

    # Power iteration
    # X = ones(nA*nB,1)./(nA*nB)
    # X = X./norm(X,1)

    # X = ones(nB,nA)
    # X = X./sum(X)

    X = Xstart
    X = np.divide(X, sum(X))

    for i in range(iters):
        print("iteration $i started")
        # X = gam1*B*X*A' + gam2*Eb*X*A' + gam2*B*X*Ea' + gam3*Eb*X*Ea'
        X = gam1 @ B @ X @ A.conj() + gam2 @ Eb @ X @ A.conj() + gam2 @ B @ X @ Ea + gam3 @ Eb @ X @ Ea
        # X = M*X
        X = np.divide(X, sum(X))

    # X  without reason again here?
    Xmat = X.reshape(nB, nA)

    # Run Hungarian method
    # Xmat = Xmat'
    # ej = munkres(-Xmat)
    # ei = 1:length(ej)
    # ids = find(ej)
    # ej = ej[ids]
    # ei = ei[ids]

    # or bipartite matching
    Xmat = Xmat.conj()
    ei, ej = bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching(scipy.sparse.csc_matrix(Xmat)))

    MATCHING = scipy.sparse.csc_matrix(ei, ej, 1, nA, nB)
    weight = X[:].conj() @ MATCHING.conj()[:]

    Ai = A[ei, ei]
    Bi = B[ej, ej]

    conserved_edges = np.nnz(Ai * Bi) / 2

    return (ei, ej, Xmat, weight, conserved_edges)