import logging
import numpy as np
import time

import pandas as pd

from . import pltm_c as model
from .util import (
    detect_input,
    read_corpus,
    perplexity,
    assign_random_topic,
    draw_phi,
    draw_theta,
    get_topics,
    Progress
)


class PLTM():

    def __init__(self,
                 *corpuses,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 n_iter=1000,
                 report_every=100):

        self.K = K
        self.alpha = alpha

        self.T = len(corpuses)
        self.W, self.vocab, self.word2id = [], [], []
        self.Z = []
        self.V, self.N = [], []
        for t, corpus in enumerate(corpuses):
            corpus = detect_input(corpus)

            W, vocab, word2id = read_corpus(corpus)
            Z = assign_random_topic(W, K)

            V = len(vocab)
            N = sum(len(d) for d in W)

            self.W.append(W)
            self.vocab.append(vocab)
            self.word2id.append(word2id)
            self.Z.append(Z)
            self.V.append(V)
            self.N.append(N)

        self.D = len(self.W[0])
        self.beta = np.array([beta]*self.T)

        logging.info(f"Corpus size: {self.D} docs, {self.N[0]} words")
        logging.info(f"  Vocabuary size: {self.V[0]}")
        logging.info(f"  Number of topics: {K}")
        logging.info(f"  alpha: {alpha:.3f}")
        logging.info(f"  beta: {beta:.3f}")

        for t in range(1, self.T):
            logging.info(f"  Side Information[{t}] size: {self.N[t]} words")
            logging.info(f"    Vocabuary size: {self.V[t]}")

        self.n_kw = [np.zeros((self.K, self.V[t]), dtype=np.int32) for t in range(self.T)]  # number of word w assigned to topic k
        self.n_dk = np.zeros((self.T, self.D, self.K), dtype=np.int32)  # number of word in document d assigned to topic k
        self.n_k = np.zeros((self.T, self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((self.T, self.D), dtype=np.int32)  # number of word in document (document length)

        model.init(self.W, self.Z, self.n_kw, self.n_dk, self.n_k, self.n_d)

        logging.info("Running Gibbs sampling inference")
        logging.info(f"Number of sampling iterations: {n_iter}")

        start = time.time()
        pbar = Progress(n_iter)
        for i in range(n_iter):
            model.inference(self.W, self.Z, self.N, self.n_kw, self.n_dk, self.n_k, self.n_d, self.alpha, self.beta)
            if i % report_every == 0:
                ppl = perplexity(self.N[0], self.n_kw[0], self.n_dk[0], self.alpha, self.beta[0])
            pbar.update(ppl)
        
        elapsed = time.time() - start
        ppl = perplexity(self.N[0], self.n_kw[0], self.n_dk[0], self.alpha, self.beta[0])
        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={ppl:.3f}")

        self.__params = {}
        self.__params['theta'] = draw_theta(self.n_dk[0], self.beta[0])
        self.__params['phi'] = draw_phi(self.n_kw[0], self.alpha)
    
    def get_topics(self, topn=10):
        return get_topics(self.n_kw[0], self.vocab[0], self.beta, topn=topn)
    
    def __getitem__(self, key):
        return self.__params[key]