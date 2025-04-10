# -*- coding: utf-8 -*-
import logging
import numpy as np
import time

from . import lda_c as lda
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


class LDA():
    
    def __init__(self,
                 corpus,
                 word2id=None,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 n_iter=1000,
                 report_every=10):
        self.corpus = detect_input(corpus)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.report_every = report_every

        if word2id:
            self.word2id = {iw.split("\t")[0]: int(iw.split("\t")[1]) for iw in open(word2id).read().strip().split("\n")}
        else:
            self.word2id = {}

        self.W, self.vocab, self.word2id = read_corpus(self.corpus, word2id=self.word2id)
        self.Z = assign_random_topic(self.W, self.K)

        self.D = len(self.W)
        self.N = sum(len(d) for d in self.W)
        self.V = len(self.vocab)

        logging.info(f"Corpus: {self.D} docs, {self.N} words, {self.V} vocab.")
        logging.info(f"Number of topics: {self.K}")
        logging.info(f"alpha: {self.alpha:.3f}")
        logging.info(f"beta: {self.beta:.3f}")

        logging.info("Running Gibbs sampling inference")
        logging.info(f"Number of sampling iterations: {self.n_iter}")

        self.n_kw = np.zeros((self.K, self.V), dtype=np.int32)  # number of word w assigned to topic k
        self.n_dk = np.zeros((self.D, self.K), dtype=np.int32)  # number of words in document d assigned to topic k
        self.n_k = np.zeros((self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((self.D), dtype=np.int32)  # number of word in document (document length)

        lda.init(self.W, self.Z, self.n_kw, self.n_dk, self.n_k, self.n_d)

        progress = Progress(self.n_iter)
        start = time.time()
        for i in range(self.n_iter):
            lda.inference(self.W, self.Z, self.n_kw, self.n_dk, self.n_k, self.n_d, self.alpha, self.beta)
            if i % self.report_every == 0:
                ppl = perplexity(self.N, self.n_kw, self.n_dk, self.alpha, self.beta)
            progress.update(ppl)
        
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
