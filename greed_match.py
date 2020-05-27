import numpy as np
import scipy


def greedy_match(X):
    m, n = np.shape(X)
    N = m @ n
    x = X[:]
    minSize = min(m, n)
    usedRows = np.zeros(m)
    usedCols = np.zeros(n)

    maxList = np.zeros(minSize)
    row = np.zeros(int, minSize)
    col = np.zeros(int, minSize)
    # y = sort(x,rev=true)
    ix = np.argsort(x, 'mergesort', True)
    y = x[ix]

    matched = 1
    index = 1
    while matched <= minSize:
        ipos = ix[index]  # position in the original vectorized matrix
        jc = np.ceil(int, ipos / m)
        ic = ipos - (jc - 1) * m
        if ic == 0:
            ic = 1

        if usedRows[ic] != 1 and usedCols[jc] != 1:
            # matched;
            row[matched] = ic
            col[matched] = jc
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1

    data = np.ones(minSize)
    M = scipy.sparse.csc_matrix(row, col, data, m, n);
    return row, col, M