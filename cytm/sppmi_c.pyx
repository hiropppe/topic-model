# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import math

from scipy.sparse import lil_matrix
from tqdm import tqdm

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log as clog, log2 as clog2
from libcpp cimport bool

ctypedef np.int32_t INT_t
ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t


def sppmi(np.ndarray[FLOAT_t, ndim=2] M not None, int k, float eps):

    cdef long long V = len(M)
    cdef long long C = M.shape[1]
    cdef long long L = V*C

    cdef np.ndarray[FLOAT_t, ndim=2] Y = np.zeros((V, C), dtype=np.float32)

    cdef float N
    cdef np.ndarray[FLOAT_t, ndim=1] Nw = np.zeros(V, dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] Nc = np.zeros(C, dtype=np.float32)
    cdef float N_ij
    cdef float shifted_positive_pmi
    
    cdef long long i, j
    cdef float Nw_i, Nc_j

    N = np.sum(M)
    Nc = np.sum(M, axis=0)
    Nw = np.sum(M, axis=1)

    pbar = tqdm(total=V) 
    for i in range(V):
        Nw_i = Nw[i]
        for j in range(C):
            Nc_j = Nc[j]
            N_ij = M[i, j]
            shifted_positive_pmi = clog2(N_ij * N / Nw_i * Nc_j + eps) - clog2(k)
            if shifted_positive_pmi > 0:
                Y[i, j] = shifted_positive_pmi
        pbar.update(n=1)
    pbar.close()

    return Y


def sppmis(M not None, int k, float eps):

    cdef long long V = M.shape[0]
    cdef long long C = M.shape[1]
    cdef long long L = V*C

    Y = lil_matrix((V, C), dtype=np.float32)

    cdef float N
    cdef np.ndarray[FLOAT_t, ndim=1] Nw = np.zeros(V, dtype=np.float32)
    cdef np.ndarray[FLOAT_t, ndim=1] Nc = np.zeros(C, dtype=np.float32)
    cdef float N_ij
    cdef float shifted_positive_pmi
    
    cdef long long i, j
    cdef long long offset, end, index
    cdef float Nw_i, Nc_j
    
    N = np.sum(M)
    Nc = np.array(np.sum(M, axis=0))[0]
    Nw = np.array(np.sum(M.T, axis=0))[0]

    pbar = tqdm(total=V) 
    for i in range(V):
        Nw_i = Nw[i]
        offset, end = M.indptr[i], M.indptr[i+1]
        for j in range(offset, end):
            index = M.indices[j]
            Nc_j = Nc[index]
            N_ij = M.data[j]
            shifted_positive_pmi = clog2(N_ij * N / Nw_i * Nc_j + eps) - clog2(k) 
            if shifted_positive_pmi > 0:
                Y[i, index] = shifted_positive_pmi
        pbar.update(n=1)
    pbar.close()

    return Y.tocsc()
