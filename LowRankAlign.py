import scipy
from scipy.sparse.linalg import eigs
import file1
from scipy.sparse import csc_matrix

def LowRankAlign(A, B, k):
    Aeigs = eigs(A, k)
    Beigs = eigs(B, k)

    V = Aeigs[0]
    U = Beigs[0]

    # can do a better job with using the eigen values instead?
    M = A @ V @ U.conj() @ B
    ei, ej = file1.edge_list(file1.bipartite_matching(scipy.sparse.csc_matrix(M)))
    return ei, ej