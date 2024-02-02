# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np
import numpy.random as npr

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX

ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t


def gibbs(list W,
          list Z,
          int N,
          np.ndarray[INT_t, ndim=2] n_wk not None,
          np.ndarray[INT_t, ndim=2] n_dk not None,
          np.ndarray[INT_t, ndim=1] n_k not None,
          np.ndarray[INT_t, ndim=1] n_b not None,
          np.ndarray[INT_t, ndim=1] n_w not None,
          np.ndarray[INT_t, ndim=1] n_ws not None,
          double alpha,
          double beta,
          double eta,
          double a0,
          double a1,
          int iter):
    cdef int D = len(W)
    cdef int K = n_dk.shape[1]
    cdef int V = n_wk.shape[0]
    cdef int n, d, k, Nd
    cdef Py_ssize_t w_dn, z_dn, z_new
    cdef double s, q, p0, p1, total
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(K)
    cdef np.ndarray[INT_t, ndim=1] seq = np.zeros(D, dtype=np.int32)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands

    # shuffle document order
    for d in range(D):
        seq[d] = d
    np.random.shuffle(seq)

    for d in range(D):
        d = seq[d]
        Nd = len(W[d])
        rands = npr.rand(2*N)
        for n in range(Nd):
            w_dn = W[d][n]
            z_dn = Z[d][n]
            s = 0.0

            # decrement
            if iter > 0:
                if z_dn == 0:
                    n_b[d] -= 1
                    # global statistics
                    n_w[w_dn] -= 1
                    n_ws[0] -= 1
                else:
                    z_dn = z_dn - 1
                    n_dk[d, z_dn] -= 1
                    n_wk[w_dn, z_dn] -= 1
                    n_k[z_dn] -= 1

            # compute p(x=0)
            if iter > 0:
                q = (a0 + n_b[d]) / (a0 + a1 + Nd - 1)
            else:
                q = 0.5

            # p(z=k,w|x=1) = p(w|z=k)p(z=k)
            for k in range(K):
                p[k] = (n_dk[d, k] + alpha) / (Nd - 1 - n_b[d] + K * alpha) * \
                       (n_wk[w_dn, k] + beta) / (n_k[k] + V * beta)
                # p(w|x=1)
                s += p[k]

            # choice between p(x=0, w) <-> p(x=1,w)
            p0 = q * (n_w[w_dn] + eta) / (n_ws[0] + V * eta)
            p1 = (1 - q) * s

            if rands[n] < (p0 / (p0 + p1)):
                z_new = 0
            else:
                total = 0.0
                z_new = 0
                for k in range(K):
                    total += p[k]
                    if rands[Nd + n] * s < total:
                        z_new = 1 + k
                        break
            Z[d][n] = z_new

            # increment
            if z_new == 0:
                n_b[d] += 1
                n_w[w_dn] += 1
                n_ws[0] += 1
            else:
                z_new = z_new - 1
                n_dk[d, z_new] += 1
                n_wk[w_dn, z_new] += 1
                n_k[z_new] += 1
