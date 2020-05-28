import os
import pickle

import numpy as np
from networkx.algorithms import bipartite

import AlignNetworks_EigenAlign
from Low_rank_Eigen import parse_args
from erdos_reyni_functions import evaluate_erdosreyni_experiment


def edgelist_to_adjmatrix(edgeList_file):
    true_alignments=np.loadtxt(edgeList_file)
    n = int(np.amax(true_alignments))+1
    e = np.shape(true_alignments)[0]
    a = np.zeros((n,n),dtype=int)
    #
    # make adjacency matrix A1
    for i in range(e):
        n1 = int(true_alignments[i, 0])#+1
        n2 = int(true_alignments[i, 1])#+1
        a[n1, n2] = 1.0
        a[n2, n1] = 1.0
    return a


def main(temp1=None):
    G1=edgelist_to_adjmatrix("data/edges_1.txt")
    G2=edgelist_to_adjmatrix("data/arenas_orig.txt")
    Matching = AlignNetworks_EigenAlign.align_networks_eigenalign(G1, G2, 8, "lowrank_svd_union", 3) # alignment step.
    print((Matching))
    Tempxx=(Matching[0])
    dd=len(Tempxx)

    split1 = np.zeros(len(Tempxx),int)
    split2 = np.zeros(len(Tempxx),int)
    for i in range(dd):
        tempMatching=Tempxx.pop()
        split1[i]=int(tempMatching[0])
        split2[i]=int(tempMatching[1])
        # recovery of correctly aligned nodes (as defined in paper):
    recovery = evaluate_erdosreyni_experiment(G1,G2,temp1,temp2)
    print(recovery)
if __name__ == "__main__":
	main()




























