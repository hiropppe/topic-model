import logging
import numpy as np
import time

import pandas as pd

from . import pltm_c as model

from gensim import matutils
from tqdm import tqdm


def train(corpus, K, alpha,  beta, n_iter):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    T, D, W, word2id = load_corpus(corpus)
    Tn = len(T)
    L = [sum(len(d) for d in D[t]) for t in T]
    Z = assign_random_topic(T, D, K)
    N = len(D[0])
    V = [len(W[t]) for t in T]

    logging.info("Corpus size: {:d} docs, {:s} words".format(N, str(L)))
    logging.info("Vocabuary size: {:s}".format(str(V)))
    logging.info("Number of topics: {:d}".format(K))
    logging.info("alpha: {:.3f}".format(alpha))
    logging.info("beta: {:.3f}".format(beta))

    n_tkw = [np.zeros((K, V[t]), dtype=np.int32) for t in T]  # number of word w assigned to topic k
    n_tdk = np.zeros((Tn, N, K), dtype=np.int32)  # number of words in document d assigned to topic k
    n_tk = np.zeros((Tn, K), dtype=np.int32)  # total number of words assigned to topic k
    n_td = np.zeros((Tn, N), dtype=np.int32)  # number of word in document (document length)

    model.init(D, Z, n_tkw, n_tdk, n_tk, n_td)
    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()
    for i in tqdm(range(n_iter)):
        model.inference(D, Z, L, n_tkw, n_tdk, n_tk, n_td, alpha, beta)
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec".format(elapsed))
    save(K, W[1], n_tkw[1], prefix='test')


def load_corpus(corpus):
    logging.info("Reading topic modeling corpus: {:s}".format(corpus))
    df = pd.read_csv(corpus)[["lang", "ctext"]]
    df.dropna(inplace=True)

    T = list(range(len(df.columns)))
    D = [[] for _ in T]
    W = [[] for _ in T]
    word2id = [{} for _ in T]
    pbar = tqdm(total=len(df))
    for row in df.iterrows():
        values = row[1].values
        for t in T:
            doc = values[t].split()
            id_doc = []
            for word in doc:
                if word not in word2id[t]:
                    word2id[t][word] = len(W[t])
                    W[t].append(word)
                id_doc.append(word2id[t][word])
            D[t].append(np.array(id_doc, dtype=np.int32))
        pbar.update(n=1)

    W[t] = np.array(W[t], dtype=np.unicode_)

    return T, D, W, word2id


def assign_random_topic(T, D, K):
    logging.info("Randomly initializing topic assignments ...")
    Z = [[] for _ in T]
    for t in T:
        for d in D[t]:
            Z[t].append(np.random.randint(K, size=len(d)))
    return Z


def save(K, W, n_kw, prefix, output_dir='./', topn=20):
    logging.info("Writing output from the last sample ...")
    logging.info("Number of top topical words: {:d}".format(topn))
    save_top_topical_words(K, W, n_kw, topn)


def save_top_topical_words(K, W, n_kw, topn):
    for k in range(K):
        topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
        print(' '.join(W[topn_indices]))
