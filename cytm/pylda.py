import logging
import numpy as np
import time

from gensim import matutils
from gensim.models.word2vec import LineSentence

from tqdm import tqdm


def train(corpus, K, alpha,  beta, n_iter):
    lda = LDA(corpus, K, alpha, beta)
    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()
    for i in tqdm(range(n_iter)):
        lda.inference()
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec".format(elapsed))

    lda.save("test")


class LDA():

    def __init__(self, corpus, K, alpha, beta):
        self.corpus = LineSentence(corpus)
        self.word2id = {}
        self.docs = []
        self.K = K
        self.W = 0
        self.n_words = 0

        tmp_vocab = []

        logging.info("Reading topic modeling corpus: {:s}".format(corpus))
        for doc in self.corpus:
            id_doc = []
            for word in doc:
                if word not in self.word2id:
                    self.word2id[word] = len(tmp_vocab)
                    tmp_vocab.append(word)
                id_doc.append(self.word2id[word])
            self.docs.append(id_doc)

        self.N = len(self.docs)
        self.vocab = np.array(tmp_vocab)
        self.V = len(self.vocab)

        self.n_kw = np.zeros((self.K, self.V))  # number of word w assigned to topic k
        self.n_k = [0] * self.K  # total number of words assigned to topic k
        self.n_dk = np.zeros((self.N, self.K))  # number of words in document d assigned to topic k
        self.n_d = [0] * self.N  # number of word in document (document length)

        self.phi = np.array([1/self.K for _ in range(self.K)])

        logging.info("Randomly initializing topic assignments ...")
        self.z = []
        for d, doc in enumerate(self.docs):
            self.W += len(doc)
            self.n_d[d] = len(doc)
            self.z.append(np.random.randint(self.K, size=self.n_d[d]))
            for n, w_dn in enumerate(doc):
                z_dn = self.z[d][n]
                self.n_dk[d, z_dn] += 1
                self.n_kw[z_dn, w_dn] += 1
                self.n_k[z_dn] += 1

        self.alpha = alpha
        self.beta = beta

        logging.info("Corpus size: {:d} docs, {:d} words".format(self.N, self.V))
        logging.info("Vocabuary size: {:d}".format(self.V))
        logging.info("Number of topics: {:d}".format(self.K))
        logging.info("alpha: {:.3f}".format(self.alpha))
        logging.info("beta: {:.3f}".format(self.beta))

    def inference(self):
        p = np.zeros(self.K, dtype=np.float64)
        seq = np.zeros(self.N, dtype=np.int32)

        for d in range(self.N):
            seq[d] = d
        np.random.shuffle(seq)

        for i in range(self.N):
            d = seq[i]
            rands = np.random.rand(self.n_d[d])
            for n in range(self.n_d[d]):
                z_dn = self.z[d][n]
                w_dn = self.docs[d][n]

                self.n_kw[z_dn, w_dn] -= 1
                self.n_dk[d, z_dn] -= 1
                self.n_k[z_dn] -= 1

                total = 0.0
                for k in range(self.K):
                    p[k] = self.n_kw[k, w_dn] + self.beta / (self.n_k[k] + self.V * self.beta) \
                           * (self.n_dk[d, k] + self.alpha)
                    total += p[k]

                rands[n] = total * rands[n]
                total = 0.0
                z_new = 0
                for k in range(self.K):
                    total += p[k]
                    if rands[n] < total:
                        z_new = k
                        break

                self.n_kw[z_new, w_dn] += 1
                self.n_dk[d, z_new] += 1
                self.n_k[z_new] += 1

    def save(self, prefix, output_dir='./', topn=20):
        logging.info("Writing output from the last sample ...")
        logging.info("Number of top topical words: {:d}".format(topn))
        self.save_top_topical_words(topn)

    def save_top_topical_words(self, topn):
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=topn, reverse=True)
            print(' '.join(self.vocab[topn_indices]))
