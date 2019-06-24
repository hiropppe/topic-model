# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX

ctypedef np.int32_t INT_t
ctypedef np.int64_t LONG_t
ctypedef np.float64_t DOUBLE_t


cdef extern from "math.h":
    double pow(double x, double y)


def init(list W,
         list X,
         list Z,
         list Y,
         np.ndarray[INT_t, ndim=2] n_kw not None,
         np.ndarray[INT_t, ndim=2] m_kx not None,
         np.ndarray[INT_t, ndim=2] n_dk not None,
         np.ndarray[INT_t, ndim=2] m_dk not None,
         np.ndarray[INT_t, ndim=1] n_k not None,
         np.ndarray[INT_t, ndim=1] m_k not None,
         np.ndarray[INT_t, ndim=1] n_d not None,
         np.ndarray[INT_t, ndim=1] m_d not None):

    cdef int N = n_dk.shape[0]
    cdef Py_ssize_t w_dn, z_dn, x_dm, y_dm
    #cdef int w_dn, z_dn, x_dm, y_dm
    cdef int d, n, m
    #cdef np.ndarray[INT_t, ndim=1] Zd
    #cdef np.ndarray[INT_t, ndim=1] Yd
    for d in range(N):
        n_d[d] = len(W[d])
        Zd = Z[d]
        for n in range(n_d[d]):
            w_dn = W[d][n]
            z_dn = Z[d][n]
            print(d, n)
            n_kw[z_dn, w_dn] += 1
            print(n_kw[z_dn, w_dn])
            n_dk[d, z_dn] += 1
            print(n_dk[d, z_dn])
            n_k[z_dn] += 1
            print(n_k[z_dn])
        m_d[d] = len(X[d])
        Yd = Y[d]
        for m in range(m_d[d]):
            x_dm = X[d][m]
            y_dm = Y[d][m]
            m_kx[y_dm, x_dm] += 1
            m_dk[d, y_dm] += 1
            m_k[y_dm] += 1



def inference(list W,
              list X,
              list Z,
              list Y,
              int Lw,
              int Lx,
              np.ndarray[INT_t, ndim=2] n_kw not None,
              np.ndarray[INT_t, ndim=2] m_kx not None,
              np.ndarray[INT_t, ndim=2] n_dk not None,
              np.ndarray[INT_t, ndim=2] m_dk not None,
              np.ndarray[INT_t, ndim=1] n_k not None,
              np.ndarray[INT_t, ndim=1] m_k not None,
              np.ndarray[INT_t, ndim=1] n_d not None,
              np.ndarray[INT_t, ndim=1] m_d not None,
              double alpha,
              double beta,
              double gamma):

    cdef int N = n_dk.shape[0]
    cdef int K = n_dk.shape[1]
    cdef int V = n_kw.shape[1]
    cdef int S = n_kw.shape[1]
    cdef int i, d, n, m
    cdef int w, x,
    cdef Py_ssize_t w_dn, z_dn, z_new, x_dm, y_dm, y_new
    cdef double total
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(K)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands_w, rands_x

    rands_w = np.random.rand(Lw)
    rands_x = np.random.rand(Lx)
    w = 0
    x = 0
    for i in range(N):
        d = i
        for n in range(n_d[d]):
            z_dn = Z[d][n]
            w_dn = W[d][n]

            n_kw[z_dn, w_dn] -= 1
            n_dk[d, z_dn] -= 1
            n_k[z_dn] -= 1
            total = 0.0

            if n_dk[d, z_dn] == 0 and m_dk[d, z_dn] > 0:
                z_new = z_dn
            else:
                for k in range(K):
                    p[k] = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta) * (n_dk[d, k] + alpha) * pow((n_dk[d, k]+1)/n_dk[d, k], m_dk[d, k])
                    total += p[k]

                rands_w[w] = total * rands_w[w]
                total = 0.0
                z_new = 0
                for k in range(K):
                    total += p[k]
                    if rands_w[w] < total:
                        z_new = k
                        break

            Z[d][n] = z_new
            n_kw[z_new, w_dn] += 1
            n_dk[d, z_new] += 1
            n_k[z_new] += 1

            w+=1

        for m in range(m_d[d]):
            y_dm = Y[d][m]
            x_dm = X[d][m]

            m_kx[y_dm, x_dm] -= 1
            m_dk[d, y_dm] -= 1
            m_k[y_dm] -= 1
            total = 0.0
            for k in range(K):
                p[k] = (m_kx[k, x_dm] + gamma) / (m_k[k] + S * gamma) * (m_dk[d, k] + gamma)
                total += p[k]

            rands_x[x] = total * rands_x[x]
            total = 0.0
            y_new = 0
            for k in range(K):
                total += p[k]
                if rands_x[x] < total:
                    y_new = k
                    break
            
            Y[d][m] = y_new
            m_kx[y_new, x_dm] += 1
            m_dk[d, y_new] += 1
            m_k[y_new] += 1

            x+=1

