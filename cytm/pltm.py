import logging
import numpy as np
import time

import pandas as pd

from . import pltm_c as model
from . import util

from gensim import matutils
from tqdm import tqdm


def train(corpus, K, alpha, beta, n_iter, report_every=100):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    T, D, W, word2id = load_corpus(corpus)
    Tn = len(T)
    L = [sum(len(d) for d in D[t]) for t in T]
    Z = assign_random_topic(T, D, K)
    N = len(D[0])
    V = [len(W[t]) for t in T]

    if len(beta) == 1:
        beta = [beta[0] for _ in T]
    beta = np.array(beta)

    if len(beta) != len(T):
        raise ValueError("number of parameter betas and document types does not match.")

    logging.info("Corpus size: {:d} docs, {:s} words".format(N, str(L)))
    logging.info("Vocabuary size: {:s}".format(str(V)))
    logging.info("Number of topics: {:d}".format(K))
    logging.info("alpha: {:.3f}".format(alpha))
    logging.info("beta: {:s}".format(str(beta)))

    n_tkw = [np.zeros((K, V[t]), dtype=np.int32) for t in T]  # number of word w assigned to topic k
    n_tdk = np.zeros((Tn, N, K), dtype=np.int32)  # number of word in document d assigned to topic k
    n_tk = np.zeros((Tn, K), dtype=np.int32)  # total number of words assigned to topic k
    n_td = np.zeros((Tn, N), dtype=np.int32)  # number of word in document (document length)

    model.init(D, Z, n_tkw, n_tdk, n_tk, n_td)
    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()
    pbar = tqdm(range(n_iter))
    for i in pbar:
        model.inference(D, Z, L, n_tkw, n_tdk, n_tk, n_td, alpha, beta)
        if i % report_every == 0:
            pbar.set_postfix(ppl="{:.3f}".format(util.ppl(L[0], n_tkw[0], n_tk[0], n_tdk[0], n_td[0], alpha, beta[0])))
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f}".format(elapsed, util.ppl(L[0], n_tkw[0], n_tk[0], n_tdk[0], n_td[0], alpha, beta[0])))
    save(K, W, n_tkw, prefix='test')


def load_corpus(corpus):
    logging.info("Reading topic modeling corpus: {:s}".format(corpus))
    df = pd.read_csv(corpus)
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

    for t in T:
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


def save_top_topical_words(K, W, n_tkw, topn):
    for t in range(len(W)):
        print("T={:d}".format(t))
        for k in range(K):
            topn_indices = matutils.argsort(n_tkw[t][k], topn=topn, reverse=True)
            print("  K={:d} {:s}".format(k, ' '.join(W[t][topn_indices])))