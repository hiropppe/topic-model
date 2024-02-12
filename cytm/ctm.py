import logging
import numpy as np
import time

import pandas as pd

from . import ctm_c as model
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


class CTM():

    def __init__(self,
                 corpus,
                 side_information,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 gamma=0.01,
                 n_iter=1000,
                 report_every=100):
        corpus = detect_input(corpus)
        side_information = detect_input(side_information)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.W, self.vocab, self.word2id = read_corpus(corpus)
        self.X, self.side_vocab, self.side2id = read_corpus(side_information)
        self.Z = assign_random_topic(self.W, self.K)
        self.Y = assign_random_topic(self.X, self.K)

        self.D = len(self.W)
        self.V, self.S = len(self.vocab), len(self.side_vocab)
        self.N, self.M = sum(len(d) for d in self.W), sum(len(d) for d in self.X)

        logging.info(f"Corpus size: {self.D} docs, {self.N} words, {self.M} side information words")
        logging.info(f"Vocabuary size: {self.V}")
        logging.info(f"Side information size: {self.S}")
        logging.info(f"Number of topics: {self.K}")

        logging.info(f"alpha: {self.alpha:.3f}")
        logging.info(f"beta: {self.beta:.3f}")
        logging.info(f"gamma: {self.gamma:.3f}")

        self.n_kw = np.zeros((self.K, self.V), dtype=np.int32)  # number of word w assigned to topic k
        self.n_dk = np.zeros((self.D, self.K), dtype=np.int32)  # number of word in document d assigned to topic k
        self.n_k = np.zeros((self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((self.D), dtype=np.int32)  # number of word in document (document length)
        self.m_kx = np.zeros((self.K, self.S), dtype=np.int32)  # number of aux word x assigned to topic k
        self.m_dk = np.zeros((self.D, self.K), dtype=np.int32)  # number of aux word in document d assigned to topic k
        self.m_k = np.zeros((self.K), dtype=np.int32)  # total number of aux words assigned to topic k
        self.m_d = np.zeros((self.D), dtype=np.int32)  # number of aux word in document (document length)

        model.init(self.W, self.X, self.Z, self.Y, self.n_kw, self.m_kx, self.n_dk, self.m_dk, self.n_k, self.m_k, self.n_d, self.m_d)

        logging.info("Running Gibbs sampling inference: ")
        logging.info(f"Number of sampling iterations: {n_iter}")

        start = time.time()
        pbar = Progress(n_iter)
        for i in range(n_iter):
            model.inference(self.W, self.X, self.Z, self.Y, self.N, self.M,
                            self.n_kw, self.m_kx, self.n_dk, self.m_dk, self.n_k, self.m_k, self.n_d, self.m_d,
                            self.alpha, self.beta, self.gamma)
            if i % report_every == 0:
                ppl = perplexity(self.N, self.n_kw, self.n_dk, self.alpha, self.beta)
            pbar.update(ppl)

        elapsed = time.time() - start
        ppl = perplexity(self.N, self.n_kw, self.n_dk, self.alpha, self.beta)
        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={ppl:.3f}")
    
        self.__params = {}
        self.__params['theta'] = draw_theta(self.n_dk, self.alpha)
        self.__params['phi'] = draw_phi(self.n_kw, self.beta)

    def get_topics(self, topn=10):
        return get_topics(self.n_kw, self.vocab, self.beta, topn=topn)
    
    def __getitem__(self, key):
        return self.__params[key]