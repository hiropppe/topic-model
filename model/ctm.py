import logging
import numpy as np
import time

import pandas as pd

from . import ctm_c as model
from . import util

from gensim import matutils
from tqdm import tqdm


def train(corpus, K, alpha, beta, gamma, n_iter, report_every=100):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    W, X, V, S = load_corpus(corpus)
    Lv, Ls = len(V), len(S)
    Lw, Lx = sum(len(d) for d in W), sum(len(d) for d in X)
    Z, Y = assign_random_topic(W, X, K)
    N = len(W)

    logging.info("Corpus size: {:d} docs, {:s} words".format(N, str(Lw)))
    logging.info("Vocabuary size: {:d}".format(Lv))
    logging.info("Number of topics: {:d}".format(K))
    logging.info("alpha: {:.3f}".format(alpha))
    logging.info("beta: {:.3f}".format(beta))
    logging.info("gamma: {:.3f}".format(gamma))

    n_kw = np.zeros((K, Lv), dtype=np.int32)  # number of word w assigned to topic k
    n_dk = np.zeros((N, K), dtype=np.int32)  # number of word in document d assigned to topic k
    n_k = np.zeros((K), dtype=np.int32)  # total number of words assigned to topic k
    n_d = np.zeros((N), dtype=np.int32)  # number of word in document (document length)
    m_kx = np.zeros((K, Ls), dtype=np.int32)  # number of aux word x assigned to topic k
    m_dk = np.zeros((N, K), dtype=np.int32)  # number of aux word in document d assigned to topic k
    m_k = np.zeros((K), dtype=np.int32)  # total number of aux words assigned to topic k
    m_d = np.zeros((N), dtype=np.int32)  # number of aux word in document (document length)

    model.init(W, X, Z, Y, n_kw, m_kx, n_dk, m_dk, n_k, m_k, n_d, m_d)
    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()
    pbar = tqdm(range(n_iter))
    for i in tqdm(range(n_iter)):
        model.inference(W, X, Z, Y, Lw, Lx, n_kw, m_kx, n_dk, m_dk, n_k, m_k, n_d, m_d, alpha, beta, gamma)
        if i % report_every == 0:
            pbar.set_postfix(ppl="{:.3f}".format(util.ppl(Lw, n_kw, n_k, n_dk, n_d, alpha, beta)))
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec".format(elapsed))
    save(K, V, S, n_kw, m_kx, prefix='test')


def load_corpus(corpus):
    logging.info("Reading topic modeling corpus: {:s}".format(corpus))
    df = pd.read_csv(corpus)
    df.dropna(inplace=True)

    W, X, V, S = [], [], [], []
    word2id, aux2id = {}, {}
    pbar = tqdm(total=len(df))
    for row in df.iterrows():
        values = row[1].values
        id_doc, id_aux = [], []
        for w in values[0].split():
            if w not in word2id:
                word2id[w] = len(V)
                V.append(w)
            id_doc.append(word2id[w])
        W.append(np.array(id_doc, dtype=np.int32))

        for x in values[1].split():
            if x not in aux2id:
                aux2id[x] = len(S)
                S.append(x)
            id_aux.append(aux2id[x])
        X.append(np.array(id_aux, dtype=np.int32))
        pbar.update(n=1)

    V = np.array(V, dtype=np.unicode_)
    S = np.array(S, dtype=np.unicode_)

    return W, X, V, S


def assign_random_topic_(W, K):
    logging.info("Randomly initializing topic assignments ...")
    Z = []
    for w in W:
        #Z.append(np.random.randint(K, size=len(w), dtype=np.int32).tolist())
        Z.append(np.random.randint(K, size=len(w)))
    return Z


def assign_random_topic(W, X, K):
    logging.info("Randomly initializing topic assignments ...")
    Z, Y = [], []
    for i in range(len(W)):
        Z.append(np.random.randint(K, size=len(W[i]), dtype=np.int32).tolist())
        Y.append(np.random.randint(K, size=len(X[i]), dtype=np.int32).tolist())
    return Z, Y


def save(K, V, S, n_kw, m_kx, prefix, output_dir='./', topn=20):
    logging.info("Writing output from the last sample ...")
    logging.info("Number of top topical words: {:d}".format(topn))
    save_top_topical_words(K, V, S, n_kw, m_kx, topn)


def save_top_topical_words(K, V, S, n_kw, m_kx, topn):
    for k in range(K):
        topn_vocab_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
        topn_aux_indices = matutils.argsort(m_kx[k], topn=topn, reverse=True)
        print("K={:d}".format(k))
        print("  word: {:s}".format(' '.join(V[topn_vocab_indices])))
        print("  aux: {:s}".format(' '.join(S[topn_aux_indices])))
