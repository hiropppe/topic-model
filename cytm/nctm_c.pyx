# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

import numpy as np

cimport numpy as np
cimport cython

ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

cdef extern from "math.h":
    double pow(double x, double y)


def init(list W,
         list X,
         list Z,
         list Y,
         list R,
         np.ndarray[INT_t, ndim=2] n_kw not None,
         np.ndarray[INT_t, ndim=2] m_kx not None,
         np.ndarray[INT_t, ndim=2] m_rx not None,
         np.ndarray[INT_t, ndim=2] n_dk not None,
         np.ndarray[INT_t, ndim=2] m_dk not None,
         np.ndarray[INT_t, ndim=2] m_dr not None,
         np.ndarray[INT_t, ndim=1] n_k not None,
         np.ndarray[INT_t, ndim=1] m_k not None,
         np.ndarray[INT_t, ndim=1] m_r not None,
         np.ndarray[INT_t, ndim=1] n_d not None,
         np.ndarray[INT_t, ndim=1] m_d not None):

    cdef int N = n_dk.shape[0]
    cdef Py_ssize_t w_dn, z_dn, x_dm, y_dm, r_dm
    cdef int d, n, m

    for d in range(N):
        n_d[d] = len(W[d])
        for n in range(n_d[d]):
            w_dn = W[d][n]
            z_dn = Z[d][n]
            n_kw[z_dn, w_dn] += 1
            n_dk[d, z_dn] += 1
            n_k[z_dn] += 1
        m_d[d] = len(X[d])
        for m in range(m_d[d]):
            x_dm = X[d][m]
            y_dm = Y[d][m]
            r_dm = R[d][m]
            if r_dm == 1:
                m_kx[y_dm, x_dm] += 1
                m_dk[d, y_dm] += 1
                m_k[y_dm] += 1
            m_rx[r_dm, x_dm] += 1
            m_dr[d, r_dm] += 1
            m_r[r_dm] += 1


def inference(list W,
              list X,
              list Z,
              list Y,
              list R,
              int Lw,
              int Lx,
              np.ndarray[INT_t, ndim=2] n_kw not None,
              np.ndarray[INT_t, ndim=2] m_kx not None,
              np.ndarray[INT_t, ndim=2] m_rx not None,
              np.ndarray[INT_t, ndim=2] n_dk not None,
              np.ndarray[INT_t, ndim=2] m_dk not None,
              np.ndarray[INT_t, ndim=2] m_dr not None,
              np.ndarray[INT_t, ndim=1] n_k not None,
              np.ndarray[INT_t, ndim=1] m_k not None,
              np.ndarray[INT_t, ndim=1] m_r not None,
              np.ndarray[INT_t, ndim=1] n_d not None,
              np.ndarray[INT_t, ndim=1] m_d not None,
              double alpha,
              double beta,
              double gamma,
              double eta):

    cdef int N = n_dk.shape[0]
    cdef int K = n_dk.shape[1]
    cdef int V = n_kw.shape[1]
    cdef int S = m_kx.shape[1]
    cdef int i, d, n, m
    cdef int iw, ixy, ixr
    cdef Py_ssize_t w_dn, z_dn, z_new, x_dm, y_dm, y_new, r_dm, r_new
    cdef double total
    cdef np.ndarray[DOUBLE_t, ndim=1] p_k = np.zeros(K)
    cdef np.ndarray[DOUBLE_t, ndim=1] p_r = np.zeros(2)
    cdef np.ndarray[DOUBLE_t, ndim=1] rands_w, rands_xy, rands_xr
    cdef double epsilon = 1e-07

    rands_w = np.random.rand(Lw)
    rands_xy = np.random.rand(Lx)
    rands_xr = np.random.rand(Lx)
    iw, ixy, ixr = 0, 0, 0
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
                    p_k[k] = (n_kw[k, w_dn] + beta) / (n_k[k] + V * beta) * (n_dk[d, k] + alpha) * pow((n_dk[d, k] + 1)/(n_dk[d, k] + epsilon), m_dk[d, k])
                    total += p_k[k]

                rands_w[iw] = total * rands_w[iw]
                total = 0.0
                z_new = 0
                for k in range(K):
                    total += p_k[k]
                    if rands_w[iw] < total:
                        z_new = k
                        break

            Z[d][n] = z_new
            n_kw[z_new, w_dn] += 1
            n_dk[d, z_new] += 1
            n_k[z_new] += 1

            iw+=1

        for m in range(m_d[d]):
            y_dm = Y[d][m]
            x_dm = X[d][m]
            r_dm = R[d][m]

            if r_dm == 1:
                m_kx[y_dm, x_dm] -= 1
                m_dk[d, y_dm] -= 1
                m_k[y_dm] -= 1
                total = 0.0
                for k in range(K):
                    p_k[k] = (m_kx[k, x_dm] + gamma) / (m_k[k] + S * gamma) * n_dk[d, k]
                    total += p_k[k]

                rands_xy[ixy] = total * rands_xy[ixy]
                total = 0.0
                y_new = 0
                for k in range(K):
                    total += p_k[k]
                    if rands_xy[ixy] < total:
                        y_new = k
                        break

                Y[d][m] = y_new
                m_kx[y_new, x_dm] += 1
                m_dk[d, y_new] += 1
                m_k[y_new] += 1

            ixy+=1

        for m in range(m_d[d]):
            y_dm = Y[d][m]
            x_dm = X[d][m]
            r_dm = R[d][m]
            m_rx[r_dm, x_dm] -= 1
            m_dr[d, r_dm] -= 1
            m_r[r_dm] -= 1

            if r_dm == 1:
                m_kx[y_dm, x_dm] -= 1
                m_dk[d, y_dm] -= 1
                m_k[y_dm] -= 1
            
            p_r[0] = (m_rx[0, x_dm] + gamma) / (m_r[0] + S * gamma) * (m_dr[d, 0] + eta)
            p_r[1] = (m_kx[y_dm, x_dm] + gamma) / (m_k[y_dm] + S * gamma) * (m_dr[d, 1] + eta)
            total = p_r[0] + p_r[1]

            rands_xr[ixr] = total * rands_xr[ixr]
            total = 0.0
            r_new = 0
            for r in range(2):
                total += p_r[r]
                if rands_xr[ixr] < total:
                    r_new = r
                    break
            
            R[d][m] = r_new
            m_rx[r_new, x_dm] += 1
            m_dr[d, r_new] += 1
            m_r[r_new] += 1
            if r_new == 1:
                m_kx[y_dm, x_dm] += 1
                m_dk[d, y_dm] += 1
                m_k[y_dm] += 1

            ixr+=1
