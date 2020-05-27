import numpy as np
import spicy

import file1


def EigenAlign_nokron(A, B, s1, s2, s3, iters):
    if np.shape(A, 0) <= np.shape(B, 0):

        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper_nokron(A, B, s1, s2, s3, iters)
        return ei, ej, Xmat, weight, conserved_edges
    else:
        ei, ej, Xmat, weight, conserved_edges = EigenAlign_helper_nokron(B, A, s1, s2, s3, iters)
        return ej, ei, Xmat.conj(), weight, conserved_edges

def EigenAlign_helper_nokron(A, B, s1, s2, s3, iters):
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

    X = np.ones(nB, nA)
    X = np.divide(X, sum(X))

    # x = copy(X)
    # for i = 1:iters
    #   y = M*x
    #   x = y./norm(y)
    #   lam = x'*y
    #   @show lam
    # end
    # Xmat = reshape(x,nB,nA)
    resid = 1
    for i in range(iters):
        i = 1
        # while resid >1e-14
        print(i)
        i += 1
        # println("iteration $i started")
        # X = gam1*B*X*A' + gam2*Eb*X*A' + gam2*B*X*Ea' + gam3*Eb*X*Ea'
        Y = gam1 @ B @ X @ A.conj() + gam2 @ Eb @ X @ A.conj() + gam2 @ B @ X @ Ea + gam3 @ Eb @ X @ Ea
        X = np.divide(Y, np.norm(Y[:], 1))

        # Y = gam1*B*X*A' + gam2*Eb*X*A' + gam2*B*X*Ea + gam3*Eb*X*Ea #(Mx)
        y = Y[:]
        x = X[:]
        lam = np.divide((x.conj() @ y), (x.conj() @ x))
        resid = np.norm(y - lam[0] @ x)
        # if resid < 1e-16
        #   @show resid
        # end
    # X    just X here? why? why?wtf!
    Xmat = X.reshape(nB, nA)

    # Run Hungarian method
    # Xmat = Xmat'
    # ej = munkres(-Xmat)
    # ei = 1:length(ej)
    # ids = find(ej)
    # ej = ej[ids]
    # ei = ei[ids]

    # or bipartite matching
    # Xmat = Xmat'
    # Xmat = Xmat'
    # ej,ei = edge_list(bipartite_matching(sparse(Xmat*100)))

    ej, ei, M = file1.greedy_match(Xmat)
    ejbp, eibp = file1.edge_list(
        file1.bipartite_matching(spicy.sparse.csc_matrix(Xmat * 10 ^ (abs(np.log10(np.amax(Xmat)))))))

    MATCHING = spicy.sparse.csc_matrix(ei, ej, 1, nA, nB)
    weight = X[:].conj() @ MATCHING.conj()[:]

    Ai = A[ei, ei]
    Bi = B[ej, ej]

    conserved_edges = np.nnz(Ai * Bi) / 2

    if resid > 1e-10:
        print("residual is bigger than 1e-14,dont know how to make it error yet")

    return (ei, ej, Xmat, eibp, ejbp)