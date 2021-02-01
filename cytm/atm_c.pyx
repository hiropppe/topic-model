# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc

ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t


def init(list D,
         list A,
         list Z,
         list X,
         np.ndarray[INT_t, ndim=2] n_kw not None,
         np.ndarray[INT_t, ndim=2] n_dk not None,
         np.ndarray[INT_t, ndim=2] n_ak not None,
         np.ndarray[INT_t, ndim=1] n_k not None,
         np.ndarray[INT_t, ndim=1] n_d not None,
         np.ndarray[INT_t, ndim=1] n_a not None,
         np.ndarray[INT_t, ndim=1] n_ad not None):

    cdef int N = n_dk.shape[0]
    cdef int N_d = 0
    cdef Py_ssize_t w_dn, z_dn, x_dn, a_dn

    for d in range(N):
        n_d[d] = len(D[d])
        n_ad[d] = len(A[d])
        for n in range(n_d[d]):
            w_dn = D[d][n]
            z_dn = Z[d][n]
            x_dn = X[d][n] 
            a_dn = A[d][x_dn]
            n_kw[z_dn, w_dn] += 1
            n_dk[d, z_dn] += 1
            n_k[z_dn] += 1
            n_ak[a_dn, z_dn] += 1
            n_a[a_dn] += 1

def inference(list D,
              list A,
              list Z,
              list X,
              int L,
              np.ndarray[INT_t, ndim=2] n_kw not None,
              np.ndarray[INT_t, ndim=2] n_dk not None,
              np.ndarray[INT_t, ndim=2] n_ak not None,
              np.ndarray[INT_t, ndim=1] n_k not None,
              np.ndarray[INT_t, ndim=1] n_d not None,
              np.ndarray[INT_t, ndim=1] n_a not None,
              np.ndarray[INT_t, ndim=1] n_ad not None,
              double alpha,
              double beta):
    cdef int N = n_dk.shape[0]
    cdef int K = n_dk.shape[1]
    cdef int V = n_kw.shape[1]
    cdef int m, n, d, w, x, a, i
    cdef Py_ssize_t w_dn, z_dn, z_new
    cdef Py_ssize_t x_dn, a_dn, x_new, a_new
    cdef double total
    cdef np.ndarray[DOUBLE_t, ndim=1] rands
    cdef np.ndarray[INT_t, ndim=1] seq = np.zeros(N, dtype=np.int32)
    #cdef pair[int, int] za_new

    cdef double p_val
    cdef vector[double] P
    cdef vector[double].iterator it

    for d in range(N):
        seq[d] = d
    np.random.shuffle(seq)

    rands = np.random.rand(L)
    w = 0
    for m in range(N):
        d = seq[m]
        for n in range(n_d[d]):
            z_dn = Z[d][n]
            w_dn = D[d][n]
            x_dn = X[d][n] 
            a_dn = A[d][x_dn]

            n_kw[z_dn, w_dn] -= 1
            n_dk[d, z_dn] -= 1
            n_k[z_dn] -= 1
            n_ak[a_dn, z_dn] -= 1
            n_a[a_dn] -= 1

            #za_new = sampling(A[d], w_dn, n_d[d], n_ad[d], n_kw, n_ak, n_k, n_a, rands[w], alpha, beta)
            #z_new = za_new.first
            #x_new = za_new.second
            
            total = 0.0
            p_val = 0.0
            P.clear()
            for k in range(K):
                p_val = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta)
                for x in range(n_ad[d]):
                    a = A[d][x]
                    p_val = p_val * (n_ak[a, k] + alpha) / (n_a[a] + K * alpha)
                    P.push_back(p_val)
                    total += p_val

            rands[w] = total * rands[w]

            i = 0
            total = 0.0
            it = P.begin()
            while it != P.end():
                total += deref(it)
                if rands[w] < total:
                    break
                inc(it)
                i += 1

            z_new = i % K
            x_new = i / K
            a_new = A[d][x_new]

            Z[d][n] = z_new
            X[d][n] = x_new
            n_kw[z_new, w_dn] += 1
            n_dk[d, z_new] += 1
            n_k[z_new] += 1
            n_ak[a_new, z_new] += 1
            n_a[a_new] += 1

            w+=1


cdef pair[int, int] sampling(np.ndarray[INT_t, ndim=1] A_d,
             int w_dn,
             int n_d,
             int n_ad,
             np.ndarray[INT_t, ndim=2] n_kw,
             np.ndarray[INT_t, ndim=2] n_ak,
             np.ndarray[INT_t, ndim=1] n_k,
             np.ndarray[INT_t, ndim=1] n_a,
             double rand,
             double alpha,
             double beta):
    cdef int K = n_kw.shape[0]
    cdef int V = n_kw.shape[1]
    cdef int k, x, i, a, j
    cdef Py_ssize_t za, z_new, x_new
    cdef double total, p_val
    cdef vector[double] pp
    cdef vector[double].iterator it
#    cdef np.ndarray[DOUBLE_t, ndim=2] p = np.zeros((n_ad, K), dtype=np.float64)
#    cdef np.ndarray[DOUBLE_t, ndim=1] za_p = np.zeros(n_ad*K, dtype=np.float64)
    cdef pair[int, int] ret

    total = 0.0
    p_val = 0.0
    for k in range(K):
        #p[:, k] = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta)
        for x in range(n_ad):
            a = A_d[x]
            p_val = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta) * (n_ak[a, k] + alpha) / (n_a[a] + K * alpha)
            pp.push_back(p_val)
            #p[x, k] = p[x, k] * (n_ak[a, k] + alpha) / (n_a[a] + K * alpha)
            total += p_val
            #total += p[x, k]

    rand = total * rand
    total = 0.0
    z_new = 0
    x_new = 0

    #za_p = p.flatten()
    #for i in range(n_ad*K):
    #    total += za_p[i]
    #    if rand < total:
    #        za = i
    #        break

    i = 0
    it = pp.begin()
    while it != pp.end():
        total += deref(it)
        if rand < total:
            za = i
            break
        inc(it)
        i += 1

    z_new = za % K
    x_new = za / K

    ret.first = z_new
    ret.second = x_new

    return ret
