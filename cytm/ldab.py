# -*- coding: utf-8 -*-
import logging
import numpy as np
import numpy.random as npr
import time

from . import ldab_c as ldab
from .util import detect_input, read_corpus, Progress, norm, cnorm, rnorm

from gensim import matutils

from scipy.special import gammaln


class LDAb():

    def __init__(self,
                 corpus,
                 K,
                 alpha=0.1,
                 beta=0.01,
                 eta=0.005,
                 a0 = 2.0,
                 a1 = 1.0,
                 n_iter=1000,
                 report_every=100):
        """ LDAb: LDA with a background distribution
        Parameters
        ----------
        corpus: {str, list[list[str]], pathlib.Path}
        K: int
            number of topics in LDA
        alpha: float, optional
            Dirichlet hyperparameter on topics (default 50/K)
        beta: float, optional
            Dirichlet hyperparameter on words (default 0.01)
        eta: float, optional
            Dirichlet hyperparameter on background (default 0.005)
        test_corpus: {str, list[list[str]], pathlib.Path}, optional
        n_iter: int, optional
            number of Gibbs iterations (default 1000)
        """
        self.corpus = detect_input(corpus)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.a0 = a0
        self.a1 = a1

        self.W, self.vocab, self.word2id = read_corpus(self.corpus)
        self.Z = assign_random_topic(self.W, self.K)

        self.D = len(self.W)
        self.N = sum(len(d) for d in self.W)
        self.V = len(self.vocab)

        logging.info(f"Corpus size: {self.D} docs, {self.N} words")
        logging.info(f"Vocabuary size: {self.V}")
        logging.info(f"Number of topics: {self.K}")
        logging.info(f"alpha: {self.alpha:.3f}")
        logging.info(f"beta: {self.beta:.3f}")

        self.n_wk = np.zeros((self.V, self.K), dtype=np.int32)  # number of word w assigned to topic k
        self.n_dk = np.zeros((self.D, self.K), dtype=np.int32)  # number of words in document d assigned to topic k
        self.n_k = np.zeros(self.K, dtype=np.int32)  # total number of words assigned to topic k
        self.n_b = np.zeros(self.D, dtype=np.int32)  # 
        self.n_w = np.zeros(self.V, dtype=np.int32)  #
        self.n_ws = np.zeros(1, dtype=np.int32)      #

        logging.info("Running Gibbs sampling inference: ")
        logging.info(f"Number of sampling iterations: {n_iter}")

        ppl = 0.0
        pbar = Progress(n_iter)
        start = time.time()
        for i in range(n_iter):
            ldab.gibbs(self.W,
                       self.Z,
                       self.N,
                       self.n_wk,
                       self.n_dk,
                       self.n_k,
                       self.n_b,
                       self.n_w,
                       self.n_ws,
                       self.alpha,
                       self.beta,
                       self.eta,
                       self.a0,
                       self.a1,
                       i)

            if i % report_every == 0:
                ppl = self.perplexity()

            pbar.update(ppl)

        elapsed = time.time() - start
        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={self.perplexity():.3f}")

    def perplexity(self):
        return perplexity(self.N,
                          self.Z,
                          self.n_w,
                          self.n_dk,
                          self.n_wk,
                          self.alpha,
                          self.beta,
                          self.eta,
                          self.a0,
                          self.a1)

    def get_topics(self, topn=10):
        tmp = cnorm(self.n_wk + self.beta)
        top_topics = []
        for k in range(self.K):
            topn_indices = matutils.argsort(tmp[:, k], topn=topn, reverse=True)
            word_probs = [(self.vocab[w], tmp[w, k]) for w in topn_indices]
            top_topics.append(word_probs)
        return top_topics

    def get_document_topics(self):
        return rnorm(self.n_dk + self.alpha)

    def get_background(self, topn=10):
        tmp = norm(self.n_w + self.eta)
        topn_indices = matutils.argsort(tmp, topn=topn, reverse=True)
        word_probs = [(self.vocab[w], tmp[w]) for w in topn_indices]
        return word_probs


def assign_random_topic(D, K):
    p = 0.5  # ratio of background
    topics = []
    for doc in D:
        base = npr.binomial(1, p, size=len(doc))
        topic = base * (1 + npr.randint(K, size=len(doc)))
        topics.append(topic)
    return topics


def perplexity(N, Z, n_w, n_dk, n_wk, alpha, beta, eta, a0, a1):
    lik = polyap(Z, a0, a1) + polyab(n_w, eta) + \
          polyad(n_dk, alpha) + polyaw(n_wk, beta)
    return np.exp(-lik / N)


def polyap(Z, a0, a1):
    D = len(Z)
    lik = 0
    for n in range(D):
        L = len(Z[n])
        n0 = sum(Z[n] == 0)
        n1 = L - n0
        lik += gammaln(a0 + a1) - gammaln(L + a0 + a1) + \
               gammaln(a0 + n0) - gammaln(a0) + \
               gammaln(a1 + n1) - gammaln(a1)
    return lik


def polyab(n_w, eta):
    V = len(n_w)
    lik = gammaln(V * eta) - gammaln(V * eta + np.sum(n_w)) \
          + np.sum(gammaln(n_w + eta) - gammaln(eta))
    return lik


def polyad(n_dk, alpha):
    N = n_dk.shape[0]
    K = n_dk.shape[1]
    n_d = np.sum(n_dk, 1)
    likelihood = np.sum(gammaln(K * alpha) - gammaln(K * alpha + n_d))
    for n in range(N):
        likelihood += np.sum(gammaln(n_dk[n, :] + alpha) - gammaln(alpha))
    return likelihood


def polyaw(n_wk, beta):
    V = n_wk.shape[0]
    K = n_wk.shape[1]
    n_w = np.sum(n_wk, 0)
    likelihood = np.sum(gammaln(V * beta) - gammaln(V * beta + n_w))
    for k in range(K):
        likelihood += np.sum(gammaln(n_wk[:, k] + beta) - gammaln(beta))
    return likelihood