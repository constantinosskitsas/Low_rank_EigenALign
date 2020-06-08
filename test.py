import networkx as nx
import numpy as np
import scipy
def main(temp1=None):

    A= [0, 3, 4, 0, 3, 4, 0, 1, 3, 4, 0, 3, 4, 4]
    B =[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4]
    nU = np.shape(A)[0]
    nV = np.shape(B)[0]
    sparsek=np.zeros((nU,nV))
    weights=[0.7136, 0.7608, 0.7521, 0.6869, 0.8656, 0.3765, 0.02878, 0.7603, 0.05077, 0.743522, 0.8666, 0.9014, 0.25047, 0.01221]

    X = scipy.sparse.csc_matrix((weights, (A, B)), shape=(nU, nV)).toarray()
    G = nx.from_numpy_matrix(X)
    D = nx.algorithms.matching.max_weight_matching(G, True)
    D1= nx.algorithms.maximal_matching(G)
    print(scipy.sparse.csc_matrix((weights, (A, B)), shape=(nU, nV)))
    print(D)
    print(D1)
if __name__ == "__main__":
	main()