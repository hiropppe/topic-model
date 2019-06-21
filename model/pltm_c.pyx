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
         list n_tkw,
         np.ndarray[INT_t, ndim=3] n_tdk not None,
         np.ndarray[INT_t, ndim=2] n_tk not None,
         np.ndarray[INT_t, ndim=2] n_td not None):

    cdef np.ndarray[INT_t, ndim=2] n_kw
    cdef int T = n_tdk.shape[0]
    cdef int N = n_tdk.shape[1]
    cdef Py_ssize_t w_tdn, z_tdn
    cdef int t, d, n
    for t in range(T):
        n_kw = n_tkw[t]
        for d in range(N):
            n_td[t, d] = len(D[t][d])
            for n in range(n_td[t, d]):
                w_tdn = D[t][d][n]
                z_tdn = Z[t][d][n]
                n_kw[z_tdn, w_tdn] += 1
                n_tdk[t, d, z_tdn] += 1
                n_tk[t, z_tdn] += 1


def inference(list D,
              list Z,
              list L,
              list n_tkw,
              np.ndarray[INT_t, ndim=3] n_tdk not None,
              np.ndarray[INT_t, ndim=2] n_tk not None,
              np.ndarray[INT_t, ndim=2] n_td not None,
              double alpha,
              #double beta):
              np.ndarray[DOUBLE_t, ndim=1] beta not None):

    cdef np.ndarray[INT_t, ndim=2] n_kw
    cdef int T = n_tdk.shape[0]
    cdef int N = n_tdk.shape[1]
    cdef int K = n_tdk.shape[2]
    cdef int V
    cdef int t, i, d, n, _t
    cdef int w
    cdef double beta_t
    cdef Py_ssize_t w_tdn, z_tdn, z_new
    cdef double total, n_dk
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(K)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands
    #cdef np.ndarray[INT_t, ndim=1] seq = np.zeros(N, dtype=np.int32)

    #for d in range(N):
    #    seq[d] = d
    #np.random.shuffle(seq)

    for t in range(T):
        n_kw = n_tkw[t]
        V = n_kw.shape[1]
        rands = np.random.rand(L[t])
        w = 0
        beta_t = beta[t]
        for i in range(N):
            #d = seq[i]
            d = i
            for n in range(n_td[t, d]):
                z_tdn = Z[t][d][n]
                w_tdn = D[t][d][n]

                n_kw[z_tdn, w_tdn] -= 1
                n_tdk[t, d, z_tdn] -= 1
                n_tk[t, z_tdn] -= 1
                total = 0.0
                for k in range(K):
                    n_dk = 0.0
                    for _t in range(T):
                        n_dk += n_tdk[_t, d, k]
                    p[k] = (n_kw[k, w_tdn] + beta_t) / (n_tk[t, k] + V * beta_t) * (n_dk + alpha)
                    total += p[k]

                rands[w] = total * rands[w]
                total = 0.0
                z_new = 0
                for k in range(K):
                    total += p[k]
                    if rands[w] < total:
                        z_new = k
                        break
            
                Z[t][d][n] = z_new
                n_kw[z_new, w_tdn] += 1
                n_tdk[t, d, z_new] += 1
                n_tk[t, z_new] += 1

                w+=1
