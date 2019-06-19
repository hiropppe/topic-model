# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX

ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

cdef extern from "math.h":
    double log (double x)


def init(list D,
         list Z,
         np.ndarray[INT_t, ndim=2] n_kw not None,
         np.ndarray[INT_t, ndim=2] n_dk not None,
         np.ndarray[INT_t, ndim=1] n_k not None,
         np.ndarray[INT_t, ndim=1] n_d not None):

    cdef int N = n_dk.shape[0]
    cdef int N_d = 0
    cdef Py_ssize_t w_dn, z_dn, z_new

    for d in range(N):
        n_d[d] = len(D[d])
        for n in range(n_d[d]):
            w_dn = D[d][n]
            z_dn = Z[d][n]
            n_kw[z_dn, w_dn] += 1
            n_dk[d, z_dn] += 1
            n_k[z_dn] += 1

def inference(list D,
              list Z,
              int L,
              np.ndarray[INT_t, ndim=2] n_kw not None,
              np.ndarray[INT_t, ndim=2] n_dk not None,
              np.ndarray[INT_t, ndim=1] n_k not None,
              np.ndarray[INT_t, ndim=1] n_d not None,
              double alpha,
              double beta):
    cdef int N = n_dk.shape[0]
    cdef int K = n_dk.shape[1]
    cdef int V = n_kw.shape[1]
    cdef int i
    cdef int d
    cdef int n
    cdef int w
    cdef Py_ssize_t w_dn, z_dn
    cdef double total
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(K)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands
    #cdef np.ndarray[INT_t, ndim=1] seq = np.zeros(N, dtype=np.int32)

    #for d in range(N):
    #    seq[d] = d
    #np.random.shuffle(seq)

    rands = np.random.rand(L)
    w = 0
    for i in range(N):
        #d = seq[i]
        d = i
        for n in range(n_d[d]):
            z_dn = Z[d][n]
            w_dn = D[d][n]

            n_kw[z_dn, w_dn] -= 1
            n_dk[d, z_dn] -= 1
            n_k[z_dn] -= 1

            total = 0.0
            for k in range(K):
                p[k] = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta) * (n_dk[d, k] + alpha)
                total += p[k]

            rands[w] = total * rands[w]
            total = 0.0
            z_new = 0
            for k in range(K):
                total += p[k]
                if rands[w] < total:
                    z_new = k
                    break
            
            Z[d][n] = z_new
            n_kw[z_new, w_dn] += 1
            n_dk[d, z_new] += 1
            n_k[z_new] += 1

            w+=1
