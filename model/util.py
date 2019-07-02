import numpy as np

from scipy.special import gammaln

from gensim import matutils
from itertools import combinations


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


def coherence(wv, W, n_kw, topn=20):
    K = len(n_kw)
    scores = []
    for k in range(K):
        topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
        for x, y in combinations(topn_indices, 2):
            w_x, w_y = W[x], W[y]
            if w_x in wv and w_y in wv:
                scores.append(wv.similarity(w_x, w_y))
    return np.mean(scores)
