import numpy as np

from scipy.special import gammaln


def ppl(L, n_kw, n_k, n_dk, n_d, alpha, beta):
    likelihood = polyad(n_dk, n_d, alpha) + polyaw(n_kw, n_k, beta)
    return np.exp(-likelihood/L)


def polyad(n_dk, n_d, alpha):
    N = n_dk.shape[0]
    K = n_dk.shape[1]
    likelihood = np.sum(gammaln(K * alpha) - gammaln(K * alpha + n_d))
    for n in range(N):
        likelihood += np.sum(gammaln(n_dk[n, :] + alpha) - gammaln(alpha))
    return likelihood


def polyaw(n_kw, n_k, beta):
    K = n_kw.shape[0]
    V = n_kw.shape[1]
    likelihood = np.sum(gammaln(V * beta) - gammaln(V * beta + n_k))
    for k in range(K):
        likelihood += np.sum(gammaln(n_kw[k, :] + beta) - gammaln(beta))
    return likelihood
