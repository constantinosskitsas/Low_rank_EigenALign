import scipy
from scipy.sparse.linalg import eigs
import bipartite_Matching
from scipy.sparse import csc_matrix

def LowRankAlign(A, B, k):
    Aeigs = eigs(A, k)
    Beigs = eigs(B, k)

    V = Aeigs[0]
    U = Beigs[0]

    # can do a better job with using the eigen values instead?
    M = A @ V @ U.conj() @ B
    ei, ej = bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching(scipy.sparse.csc_matrix(M)))
    return ei, ej