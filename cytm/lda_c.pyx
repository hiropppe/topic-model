# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX

ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t


def init(list D,
         list Z,
         np.ndarray[INT_t, ndim=2] n_kw not None,
         np.ndarray[INT_t, ndim=2] n_dk not None,
         np.ndarray[INT_t, ndim=1] n_k not None,
         np.ndarray[INT_t, ndim=1] n_d not None):

    cdef int N = n_dk.shape[0]
    cdef int N_d
    cdef Py_ssize_t d, n, w_dn, z_dn
    cdef np.ndarray[INT_t, ndim=1] zd
    cdef np.ndarray[INT_t, ndim=1] dd
    for d in range(N):
        N_d = len(D[d])
        n_d[d] = N_d
        zd = Z[d]
        dd = D[d]
        for n in range(N_d):
            z_dn = zd[n]
            w_dn = dd[n]
            n_kw[z_dn, w_dn] += 1
            n_dk[d, z_dn] += 1
            n_k[z_dn] += 1


def inference(list D,
              list Z,
              np.ndarray[INT_t, ndim=2] n_kw not None,
              np.ndarray[INT_t, ndim=2] n_dk not None,
              np.ndarray[INT_t, ndim=1] n_k not None,
              np.ndarray[INT_t, ndim=1] n_d not None,
              double alpha,
              double beta):
    cdef int N = n_dk.shape[0]
    cdef int K = n_dk.shape[1]
    cdef int V = n_kw.shape[1]
    cdef int M
    cdef Py_ssize_t m, n, d, w_dn, z_dn, z_new
    cdef double total
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(K)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands
    cdef np.ndarray[INT_t, ndim=1] seq = np.zeros(N, dtype=np.int32)
    cdef np.ndarray[INT_t, ndim=1] zd
    cdef np.ndarray[INT_t, ndim=1] dd

    for d in range(N):
        seq[d] = d
    np.random.shuffle(seq)

    for m in range(N):
        d = seq[m]
        zd = Z[d]
        dd = D[d]
        M = n_d[d]
        rands = np.random.rand (M)
        for n in range(M):
            z_dn = zd[n]
            w_dn = dd[n]

            n_kw[z_dn, w_dn] -= 1
            n_dk[d, z_dn] -= 1
            n_k[z_dn] -= 1

            total = 0.0
            for k in range(K):
                p[k] = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta) * (n_dk[d, k] + alpha)
                total += p[k]

            rands[n] = total * rands[n]
            total = 0.0
            z_new = 0
            for k in range(K):
                total += p[k]
                if rands[n] < total:
                    z_new = k
                    break
            
            Z[d][n] = z_new
            n_kw[z_new, w_dn] += 1
            n_dk[d, z_new] += 1
            n_k[z_new] += 1

