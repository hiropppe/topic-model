cimport numpy as np

ctypedef np.int_t INT_t
ctypedef np.float64_t DOUBLE_t

cdef class LDA:
    cdef:
        object corpus
        dict word2id
        list docs
        int K
        int W
        int n_words
        int N
        int V
        np.ndarray[INT_t, ndim=1] vocab
        np.ndarray[INT_t, ndim=2] n_dk
        np.ndarray[INT_t, ndim=2] n_kw
        np.ndarray[INT_t, ndim=1] n_k
        np.ndarray[INT_t, ndim=1] n_d
        np.ndarray[DOUBLE_t, ndim=1] phi
        double alpha
        double beta
