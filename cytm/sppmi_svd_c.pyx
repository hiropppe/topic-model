# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import math

from tqdm import tqdm

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log as clog, log2 as clog2
from libcpp cimport bool

ctypedef np.int32_t INT_t
ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t


cdef double absolute_discounting(np.ndarray[FLOAT_t, ndim=2] M,
                                 int i,
                                 int j,
                                 float d):
    """ SMOOTHING: absolute discounting
    :param M: cooccur matrix
    :param i, j: index
    :param d: discounting value (0, 1)
    :param V: vocab. size
    :param N0: number of words count[word]==0
    :return: smoothed value
    """
    cdef int c, k
    cdef int V, N0

    if M[i, j] > 0:
        return M[i, j] - d
    else:
        V = M.shape[1]
        c = 0
        for k in range(V):
            if M[i, k] > 0:
                c += 1
        N0 = V - c
        #N0 = V - np.count_nonzero(M[i])
        return d * (V - N0) / N0


def sppmi(np.ndarray[FLOAT_t, ndim=2] M not None,
          int shift,
          bool has_abs_dis,
          bool has_cds,
          float eps):

    cdef int V = len(M)
    cdef int C = M.shape[1]
    cdef int L = V*C

    cdef np.ndarray[FLOAT_t, ndim=2] Y = np.zeros((V, C), dtype=np.float32)

    cdef float N
    cdef np.ndarray[FLOAT_t, ndim=1] Nw = np.zeros(V, dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] Nc = np.zeros(C, dtype=np.float32)
    cdef int n1, n2
    cdef float d
    cdef float N_ij
    cdef float shifted_positive_pmi
    
    cdef int i, j
    cdef float Nw_i, Nc_j
    cdef int k, report_every = L/1000

    N = np.sum(M)
    Nc = np.sum(M, axis=0)
    Nw = np.sum(M, axis=1)
    if has_abs_dis:
        n1 = L - np.count_nonzero(M-1)
        n2 = L - np.count_nonzero(M-2)
        if n1 == 0 and n2 == 0:
            d = 0
        else:
            d = n1 / (n1 + 2 * n2)

    if has_cds:
        # Context Distributional Smoothing
        N = N**0.75
        Nw = Nw**0.75
        Nc = Nc**0.75

    k = 1
    pbar = tqdm(total=L) 
    for i in range(V):
        Nw_i = Nw[i]
        for j in range(C):
            Nc_j = Nc[j]
            if has_abs_dis:
                N_ij = absolute_discounting(M, i, j, d)
            else:
                N_ij = M[i, j]

            shifted_positive_pmi = clog2(N_ij * N / Nw_i * Nc_j + eps)        
            M[i, j] = max(0, shifted_positive_pmi - clog(shift))

            if k % report_every == 0:
                pbar.update(n=report_every)
            k += 1

    return M
