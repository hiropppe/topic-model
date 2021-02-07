# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import math

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log as clog, log2 as clog2
from libcpp cimport bool

ctypedef np.int32_t INT_t
ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t


def absolute_discounting(np.ndarray[FLOAT_t, ndim=2] C not None,
                         int i,
                         int j,
                         float d):
    """ SMOOTHING: absolute discounting
    :param C: cooccur matrix
    :param i, j: index
    :param d: discounting value (0, 1)
    :param V: vocab. size
    :param N0: number of words count[word]==0
    :return: smoothed value
    """
    cdef float V, N0

    if C[i][j] > 0:
        return C[i][j] - d
    else:
        V = C.shape[1]
        N0 = V - np.count_nonzero(C[i])
        return d * (V - N0) / N0


def sppmi(np.ndarray[FLOAT_t, ndim=2] C not None,
          int shift,
          bool has_abs_dis,
          bool has_cds,
          float eps):

    cdef int V = len(C)
    cdef np.ndarray[FLOAT_t, ndim=2] M = np.zeros((V, V), dtype=np.float32)
    cdef int i, j
    cdef float shifted_positive_pmi
    cdef float N
    cdef np.ndarray[FLOAT_t, ndim=1] Nc = np.zeros(V, dtype=np.float32)
    cdef int n1, n2
    cdef float d
    cdef float Nc_i, Nc_j
    cdef float Cwc

    N = np.sum(C)
    Nc = np.sum(C, axis=0)

    if has_abs_dis:
        # compute constant value d 
        size = C.shape[0] * C.shape[1]
        n1 = size - np.count_nonzero(C-1)
        n2 = size - np.count_nonzero(C-2)
        d = n1 / (n1 + 2 * n2)
        print(f'discount value d: {d}')

    if has_cds:
        # Context Distributional Smoothing
        N = N**0.75
        Nc = Nc**0.75

    for i in range(V):
        Nc_i = Nc[i]
        for j in range(V):
            Nc_j = Nc[j]
            if has_abs_dis:
                Cwc = absolute_discounting(C, i, j, d)
            else:
                Cwc = C[i, j]
            shifted_positive_pmi = clog2(Cwc * N / Nc_j * Nc_i + eps)        
            M[i, j] = max(0, shifted_positive_pmi - clog(shift))

    return M



