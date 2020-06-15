import json
import logging
import numpy as np
import time

from . import lda_c as lda
from .util import perplexity, get_coherence_model

from gensim import matutils
from gensim.models.word2vec import LineSentence
from pathlib import Path

from tqdm import tqdm_notebook
from tqdm import tqdm

notebook = False


def train(input,
          k,
          alpha,
          beta,
          top_words=20,
          test_texts=None,
          coherence_model="u_mass",
          wv=None,
          coo_matrix=None,
          coo_word2id=None,
          n_iter=1000,
          report_every=100,
          prefix="lda",
          output_dir=".",
          from_streamlit=False,
          verbose=False):

    if isinstance(input, list):
        corpus = input
        print_cm = False
    else:
        logging.info("Reading topic modeling corpus: {:s}".format(input))
        corpus = LineSentence(input)
        print_cm = True

    D, W, word2id = read_corpus(corpus)
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

    if print_cm:
        cm = get_coherence_model(W, n_kw, top_words, coherence_model, test_texts=test_texts, corpus=corpus,
                                 coo_matrix=coo_matrix, coo_word2id=coo_word2id, verbose=verbose)
    else:
        cm = None

    logging.info("Running Gibbs sampling inference: ")
    logging.info("Number of sampling iterations: {:d}".format(n_iter))

    start = time.time()

    if notebook:
        pbar = tqdm_notebook(total=n_iter)
    else:
        pbar = tqdm(total=n_iter)

    if from_streamlit:
        import streamlit as st
        pbar_text = st.empty()
        pbar = st.progress(0)

    ppl = 0.0
    coh = 0.0
    for i in range(n_iter):
        lda.inference(D, Z, L, n_kw, n_dk, n_k, n_d, alpha, beta)

        if i % report_every == 0:
            ppl = perplexity(L, n_kw, n_k, n_dk, n_d, alpha, beta)
            if cm:
                coh = cm.score()
            else:
                coh = None

        update_progress(i, pbar, pbar_text, ppl, coh)

    elapsed = time.time() - start

    ppl = perplexity(L, n_kw, n_k, n_dk, n_d, alpha, beta)
    if cm:
        coh = cm.score()
        logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f} coh={:.3f}".format(
            elapsed, ppl, coh))
    else:
        logging.info("Sampling completed! Elapsed {:.4f} sec ppl={:.3f}".format(
            elapsed, ppl))

    save(W, Z, n_kw, n_dk, n_k, n_d, alpha, beta, prefix=prefix, output_dir=output_dir)


def update_progress(i, pbar, st_text, ppl, coh):
    if hasattr(pbar, 'progress'):
        pbar.progress(i + 1)
        if coh is not None:
            st_text.text(f'Iteration {i+1} ppl={ppl} coherence={coh}')
        else:
            st_text.text(f'Iteration {i+1} ppl={ppl}')
    else:
        pbar.update(n=1)
        if coh is not None:
            pbar.set_postfix(ppl="{:.3f}".format(ppl), coh="{:.3f}".format(coh))
        else:
            pbar.set_postfix(ppl="{:.3f}".format(ppl))


def read_corpus(corpus):
    D = []
    W, word2id = [], {}
    for doc in corpus:
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
    # save_z(Z, prefix, output_dir)
    # save_theta(n_dk, n_d, alpha, prefix, output_dir)
    # save_phi(n_kw, n_k, beta, prefix, output_dir)
    save_informative_word(W, n_kw, n_k, beta, topn, prefix, output_dir)


def topics(W, n_kw, topn):
    topics = []
    for k in range(len(n_kw)):
        topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
        topics.append([W[w] for w in topn_indices])
    return topics


def save_topic(W, n_kw, n_k, beta, topn, prefix, output_dir):
    logging.info("Writing topics ...")
    output_path = output_dir / (prefix + ".topic")
    K = len(n_kw)
    V = n_kw.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        if notebook:
            pbar = tqdm_notebook(range(K))
        else:
            pbar = tqdm(range(K))
        for k in pbar:
            topn_indices = matutils.argsort(n_kw[k], topn=topn, reverse=True)
            print(" ".join(["{:s}*{:.4f}".format(W[w], ((n_kw[k, w] + beta) /
                                                        (n_k[k] + V * beta))) for w in topn_indices]), file=fo)


def save_theta(n_dk, n_d, alpha, prefix, output_dir):
    logging.info("Writing θ ...")
    output_path = output_dir / (prefix + ".theta")
    N = len(n_dk)
    K = n_dk.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        if notebook:
            pbar = tqdm_notebook(range(N))
        else:
            pbar = tqdm(range(N))
        for d in pbar:
            print(" ".join(["{:.4f}".format((n_dk[d, k] + alpha)/(n_d[d] + K * alpha))
                            for k in range(K)]), file=fo)


def save_z(Z, prefix, output_dir):
    logging.info("Writing z ...")
    output_path = output_dir / (prefix + ".z")
    with open(output_path.as_posix(), "w") as fo:
        if notebook:
            pbar = tqdm_notebook(range(len(Z)))
        else:
            pbar = tqdm(range(len(Z)))
        for d in pbar:
            print(" ".join(Z[d].astype(np.unicode_)), file=fo)


def save_phi(n_kw, n_k, beta, prefix, output_dir):
    logging.info("Writing Φ ...")
    output_path = output_dir / (prefix + ".phi")
    K = len(n_kw)
    V = n_kw.shape[1]
    with open(output_path.as_posix(), "w") as fo:
        if notebook:
            pbar = tqdm_notebook(range(K))
        else:
            pbar = tqdm(range(K))
        for k in pbar:
            print(" ".join(["{:.4f}".format((n_kw[k, w] + beta)/(n_k[k] + V * beta))
                            for w in range(V)]), file=fo)


def save_informative_word(W, n_kw, n_k, beta, topn, prefix, output_dir):
    logging.info("Writing important words ...")
    output_path = output_dir / (prefix + ".jlh")
    K = len(n_kw)
    V = n_kw.shape[1]
    n_w = {}
    topics = []
    with open(output_path.as_posix(), "w") as fo:
        for w in range(V):
            n_w[w] = n_kw[:, w].sum()
        if notebook:
            pbar = tqdm_notebook(range(K))
        else:
            pbar = tqdm(range(K))
        jlh_scores = np.zeros((K, V), dtype=np.float32)
        for k in pbar:
            for w in range(V):
                glo = (n_kw[k, w] + beta)/(n_w[w] + V * beta)
                loc = (n_kw[k, w] + beta)/(n_k[k] + V * beta)
                jlh_scores[k, w] = (glo-loc) * (glo/loc)
            topn_informative_words = matutils.argsort(jlh_scores[k], topn=topn, reverse=True)
            # print(" ".join(["{:s}*{:.4f}".format(W[w], jlh_scores[k, w])
            #                for w in topn_informative_words]), file=fo)
            topics.append((k, [(W[w], float(jlh_scores[k, w])) for w in topn_informative_words]))
        print(json.dumps(topics), file=fo)
