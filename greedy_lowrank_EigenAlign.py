import numpy as np
import scipy
def greedy_lowrank(U1, V1):
    k = np.shape(U1, 1)
    n = np.shape(U1, 0)

    ma = np.zeros(int, n)
    mb = np.zeros(int, n)

    ma_ids = range(0, n)
    mb_ids = range(0, n)

    U1t = U1.conj()
    V1t = V1.conj()
    lastid = n

    for i in range(n):
        curmax = 0
        curma = 0
        curmb = 0
        for mi in ma_ids:
            u = U1t[:, mi]
            for mj in mb_ids:

                v = V1t[:, mj]
                m = np.dot(u, v)
                if m > curmax:
                    curmax = m
                    curma = mi
                    curmb = mj

        if curma == 0:
            lastid = i - 1
            break

        print(curma)
        print(curmb)
        ma[i] = curma
        mb[i] = curmb
        ma_ids = ma_ids.diference(curma)
        mb_ids = mb_ids.diference(curmb)

    ma = ma[1:lastid]
    mb = mb[1:lastid]
    return ma, mb
