import json
import logging
import numpy as np
import sys
import time

import pandas as pd

from . import nctm_c as model
from .util import perplexity, EmbeddingCoherence, PMICoherence

from operator import itemgetter
from gensim import matutils
from pathlib import Path


def train(corpus,
          K,
          alpha,
          beta,
          gamma,
          eta,
          wv=None,
          coo_matrix=None,
          coo_word2id=None,
          n_iter=1000,
          report_every=100,
          prefix="nctm",
          output_dir=".",
          verbose=False,
          on_notebook=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    W, X, V, S = load_corpus(corpus)
    Lv, Ls = len(V), len(S)
    Lw, Lx = sum(len(d) for d in W), sum(len(d) for d in X)
    Z, Y, R = assign_random_topic(W, X, K)
    N = len(W)

    logging.info("Corpus size: {:d} docs, {:s} words".format(N, str(Lw)))
    logging.info("Vocabuary size: {:d}".format(Lv))
    logging.info("Number of topics: {:d}".format(K))
    logging.info("alpha: {:.3f}".format(alpha))
    logging.info("beta: {:.3f}".format(beta))
    logging.info("gamma: {:.3f}".format(gamma))
    logging.info("eta: {:.3f}".format(eta))

    n_kw = np.zeros((K, Lv), dtype=np.int32)  # number of word w assigned to topic k
    n_dk = np.zeros((N, K), dtype=np.int32)  # number of word in document d assigned to topic k
    n_k = np.zeros((K), dtype=np.int32)  # total number of words assigned to topic k
    n_d = np.zeros((N), dtype=np.int32)  # number of word in document (document length)
    m_kx = np.zeros((K, Ls), dtype=np.int32)  # number of aux word x assigned to topic k
    m_dk = np.zeros((N, K), dtype=np.int32)  # number of aux word in document d assigned to topic k
    m_k = np.zeros((K), dtype=np.int32)  # total number of aux words assigned to topic k
    m_d = np.zeros((N), dtype=np.int32)  # number of aux word in document (document length)
    m_rx = np.zeros((2, Ls), dtype=np.int32)
    m_dr = np.zeros((N, 2), dtype=np.int32)
    m_r = np.zeros((2), dtype=np.int32)

    model.init(W, X, Z, Y, R, n_kw, m_kx, m_rx, n_dk, m_dk, m_dr, n_k, m_k, m_r, n_d, m_d)

    if coo_matrix is not None:
        logging.info("Initializing PMI Coherence Model...")
        coherence = PMICoherence(coo_matrix, coo_word2id, W, n_kw, topn=20)
    elif wv is not None:
        logging.info("Initialize Word Embedding Coherence Model...")
        coherence = EmbeddingCoherence(wv, V, n_kw, topn=20)
    else:
        coherence = None

    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))
    start = time.time()

    if on_notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    pbar = tqdm(range(n_iter))
    for i in pbar:
        model.inference(W, X, Z, Y, R, Lw, Lx, n_kw, m_kx, m_rx, n_dk, m_dk,
                        m_dr, n_k, m_k, m_r, n_d, m_d, alpha, beta, gamma, eta)
        if i % report_every == 0:
            ppl = perplexity(Lw, n_kw, n_k, n_dk, n_d, alpha, beta)
            if coherence:
                coh = coherence.score()
                pbar.set_postfix(ppl="{:.3f}".format(ppl), coh="{:.3f}".format(coh))
            else:
                pbar.set_postfix(ppl="{:.3f}".format(ppl))
    elapsed = time.time() - start
    ppl = perplexity(Lw, n_kw, n_k, n_dk, n_d, alpha, beta)
    if coherence:
        coh = coherence.score()
        logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f} coh={:.3f}".format(
            elapsed, ppl, coh))
    else:
        logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f}".format(
            elapsed, ppl))

    save(Z, V, S, n_kw, n_dk, n_k, n_d, m_k, m_kx, m_r, m_rx, m_dr,
         alpha, beta, gamma, eta, prefix=prefix, output_dir=output_dir)


def load_corpus(corpus):
    logging.info("Reading topic modeling corpus: {:s}".format(corpus))
    df = pd.read_csv(corpus, header=None)
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


def assign_random_topic(W, X, K):
    logging.info("Randomly initializing topic assignments ...")
    Z, Y, R = [], [], []
    for i in range(len(W)):
        Z.append(np.random.randint(K, size=len(W[i]), dtype=np.int32))
        Y.append(np.random.randint(K, size=len(X[i]), dtype=np.int32))
        R.append(np.random.randint(2, size=len(X[i]), dtype=np.int32))
    return Z, Y, R


def save(Z, V, S, n_kw, n_dk, n_k, n_d, m_k, m_kx, m_r, m_rx, m_dr, alpha, beta, gamma, eta, prefix, output_dir='./', topn=20):
    logging.info("Writing output from the last sample ...")
    logging.info("Number of top topical words: {:d}".format(topn))

    output_dir = Path(output_dir)
    save_topic(V, S, n_kw, n_k, m_kx, beta, topn, prefix, output_dir)
    save_z(Z, prefix, output_dir)
    save_theta(n_dk, n_d, alpha, prefix, output_dir)
    save_phi(n_kw, n_k, beta, prefix, output_dir)
    save_informative_word(V, n_kw, n_k, beta, topn, prefix, output_dir)
    save_rx_prob(S, m_r, m_rx, gamma, prefix, output_dir)


def save_topic(V, S, n_kw, n_k, m_kx, beta, topn, prefix, output_dir):
    output_path = output_dir / (prefix + ".topic")
    K = len(n_kw)
    Lv = n_kw.shape[1]
    with open(output_path.__str__(), "w") as fo:
        for k in range(K):
            topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
            topn_aux_indices = matutils.argsort(m_kx[k], topn=topn, reverse=True)
            print("K={:d}".format(k), file=fo)
            print("  word: {:s}".format(" ".join(["{:s}*{:.4f}".format(V[w], ((n_kw[k, w] + beta) /
                                                                              (n_k[k] + Lv * beta))) for w in topn_indices])), file=fo)
            print("  aux: {:s}".format(' '.join(S[topn_aux_indices])), file=fo)


def save_z(Z, prefix, output_dir):
    output_path = output_dir / (prefix + ".z")
    with open(output_path.as_posix(), "w") as fo:
        for d in range(len(Z)):
            print(" ".join(Z[d].astype(np.unicode_)), file=fo)


def save_theta(n_dk, n_d, alpha, prefix, output_dir):
    output_path = output_dir / (prefix + ".theta")
    N = len(n_dk)
    K = n_dk.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        for d in range(N):
            print(" ".join(["{:.4f}".format((n_dk[d, k] + alpha)/(n_d[d] + K * alpha))
                            for k in range(K)]), file=fo)


def save_phi(n_kw, n_k, beta, prefix, output_dir):
    output_path = output_dir / (prefix + ".phi")
    K = len(n_kw)
    V = n_kw.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        for k in range(K):
            print(" ".join(["{:.4f}".format((n_kw[k, w] + beta)/(n_k[k] + V * beta))
                            for w in range(V)]), file=fo)


def save_informative_word(V, n_kw, n_k, beta, topn, prefix, output_dir):
    output_path = output_dir / (prefix + ".jlh")
    K = len(n_kw)
    Lv = n_kw.shape[1]
    n_w = {}
    topics = []
    with open(output_path.as_posix(), "w") as fo:
        for w in range(Lv):
            n_w[w] = n_kw[:, w].sum()

        jlh_scores = np.zeros((K, Lv), dtype=np.float32)
        for k in range(K):
            for w in range(Lv):
                glo = (n_kw[k, w] + beta)/(n_w[w] + Lv * beta)
                loc = (n_kw[k, w] + beta)/(n_k[k] + Lv * beta)
                jlh_scores[k, w] = (glo-loc) * (glo/loc)
            topn_informative_words = matutils.argsort(jlh_scores[k], topn=topn, reverse=True)
            #print(" ".join(["{:s}*{:.4f}".format(V[w], jlh_scores[k, w])
            #                for w in topn_informative_words]), file=fo)
            topics.append((k, [(V[w], float(jlh_scores[k, w])) for w in topn_informative_words]))
        print(json.dumps(topics), file=fo)


def save_rx_prob(S, m_r, m_rx, gamma, prefix, output_dir):
    output_path = output_dir / (prefix + ".rxp")
    Ls = m_rx.shape[1]
    rxp = []
    with open(output_path.as_posix(), "w") as fo:
        for x in range(Ls):
            sum_rx = m_rx[:, x].sum()
            p_rx0 = (m_rx[0, x] + gamma) / (sum_rx + Ls * gamma)
            p_rx1 = (m_rx[1, x] + gamma) / (sum_rx + Ls * gamma)
            rxp.append((S[x], p_rx0, p_rx1))
        rxp.sort(key=itemgetter(1), reverse=True)
        for e in rxp:
            print("{:s} {:.3f} {:.3f}".format(e[0], e[1], e[2]), file=fo)
