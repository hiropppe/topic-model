import logging
import numpy as np
import time

from . import lda_c as lda
from . import util

from gensim import matutils
from gensim.models.word2vec import LineSentence
from pathlib import Path
from tqdm import tqdm


def train(corpus, k, alpha, beta, n_iter, report_every=100, prefix="lda", output_dir="."):
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
        if i % report_every == 0:
            pbar.set_postfix(ppl="{:.3f}".format(util.ppl(L, n_kw, n_k, n_dk, n_d, alpha, beta)))
    elapsed = time.time() - start
    logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f}".format(
        elapsed, util.ppl(L, n_kw, n_k, n_dk, n_d, alpha, beta)))
    save(W, Z, n_kw, n_dk, n_k, n_d, alpha, beta, prefix=prefix, output_dir=output_dir)


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


def save(W, Z, n_kw, n_dk, n_k, n_d, alpha, beta, prefix, output_dir='.', topn=20):
    logging.info("Writing output from the last sample ...")
    logging.info("Number of top topical words: {:d}".format(topn))

    output_dir = Path(output_dir)
    save_topic(W, n_kw, n_k, beta, topn, prefix, output_dir)
    save_z(Z, prefix, output_dir)
    save_theta(n_dk, n_d, alpha, prefix, output_dir)
    save_phi(n_kw, n_k, beta, prefix, output_dir)
    save_informative_word(W, n_kw, n_k, beta, topn, prefix, output_dir)


def save_topic(W, n_kw, n_k, beta, topn, prefix, output_dir):
    output_path = output_dir / (prefix + ".topic")
    K = len(n_kw)
    V = n_kw.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        for k in range(K):
            topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
            print(" ".join(["{:s}*{:.4f}".format(W[w], ((n_kw[k, w] + beta) /
                                                        (n_k[k] + V * beta))) for w in topn_indices]), file=fo)


def save_informative_word(W, n_kw, n_k, beta, topn, prefix, output_dir):
    output_path = output_dir / (prefix + ".jlh")
    K = len(n_kw)
    V = n_kw.shape[1]
    n_w = {}
    with open(output_path.as_posix(), "w") as fo:
        for w in range(V):
            n_w[w] = n_kw[:, w].sum()

        jlh_scores = np.zeros((K, V), dtype=np.float32)
        for k in range(K):
            for w in range(V):
                glo = (n_kw[k, w] + beta)/(n_w[w] + V * beta)
                loc = (n_kw[k, w] + beta)/(n_k[k] + V * beta)
                jlh_scores[k, w] = (glo-loc) * (glo/loc)
            topn_informative_words = matutils.argsort(jlh_scores[k], topn=topn, reverse=True)
            print(" ".join(["{:s}*{:.4f}".format(W[w], jlh_scores[k, w])
                            for w in topn_informative_words]), file=fo)


def save_theta(n_dk, n_d, alpha, prefix, output_dir):
    output_path = output_dir / (prefix + ".theta")
    N = len(n_dk)
    K = n_dk.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        for d in range(N):
            print(" ".join(["{:.4f}".format((n_dk[d, k] + alpha)/(n_d[d] + K * alpha))
                            for k in range(K)]), file=fo)


def save_z(Z, prefix, output_dir):
    output_path = output_dir / (prefix + ".z")
    with open(output_path.as_posix(), "w") as fo:
        for d in range(len(Z)):
            print(" ".join(Z[d].astype(np.unicode_)), file=fo)


def save_phi(n_kw, n_k, beta, prefix, output_dir):
    output_path = output_dir / (prefix + ".phi")
    K = len(n_kw)
    V = n_kw.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        for k in range(K):
            print(" ".join(["{:.4f}".format((n_kw[k, w] + beta)/(n_k[k] + V * beta))
                            for w in range(V)]), file=fo)
