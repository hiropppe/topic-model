import logging
import numpy as np
import time

from . import lda_c as lda
from . import util

from gensim import matutils
from gensim.models.word2vec import LineSentence
from tqdm import tqdm


def train(corpus, k, alpha,  beta, n_iter):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    D, W, word2id = load_corpus(corpus)
    L = sum(len(d) for d in D)
    K = k
    Z = assign_random_topic(D, K)
    N = len(D)
    V = len(W)

    logging.info("Corpus size: {:d} docs, {:d} words".format(N, L))
    logging.info("Vocabuary size: {:d}".format(V))
    logging.info("Number of topics: {:d}".format(K))
    logging.info("alpha: {:.3f}".format(alpha))
    logging.info("beta: {:.3f}".format(beta))

    n_kw = np.zeros((K, V), dtype=np.int32)  # number of word w assigned to topic k
    n_dk = np.zeros((N, K), dtype=np.int32)  # number of words in document d assigned to topic k
    n_k = np.zeros((K), dtype=np.int32)  # total number of words assigned to topic k
    n_d = np.zeros((N), dtype=np.int32)  # number of word in document (document length)

    lda.init(D, Z, n_kw, n_dk, n_k, n_d)

    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()
    pbar = tqdm(range(n_iter))
    for i in pbar:
        lda.inference(D, Z, L, n_kw, n_dk, n_k, n_d, alpha, beta)
        if i % 10 == 0:
            pbar.set_postfix(ppl="{:.3f}".format(util.ppl(L, n_kw, n_k, n_dk, n_d, alpha, beta)))
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec".format(elapsed))
    save(K, W, n_kw, prefix='test')


def load_corpus(corpus):
    logging.info("Reading topic modeling corpus: {:s}".format(corpus))
    D = []
    W, word2id = [], {}
    for doc in LineSentence(corpus):
        id_doc = []
        for word in doc:
            if word not in word2id:
                word2id[word] = len(W)
                W.append(word)
            id_doc.append(word2id[word])
        D.append(np.array(id_doc, dtype=np.int32))

    W = np.array(W, dtype=np.unicode_)

    return D, W, word2id


def assign_random_topic(D, K):
    logging.info("Randomly initializing topic assignments ...")
    Z = []
    for d in D:
        Z.append(np.random.randint(K, size=len(d)))
    return Z


def save(K, W, n_kw, prefix, output_dir='./', topn=20):
    logging.info("Writing output from the last sample ...")
    logging.info("Number of top topical words: {:d}".format(topn))
    save_top_topical_words(K, W, n_kw, topn)


def save_top_topical_words(K, W, n_kw, topn):
    for k in range(K):
        topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
        print(' '.join(W[topn_indices]))
