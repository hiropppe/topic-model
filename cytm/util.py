import logging
import numpy as np
import pathlib
import re
# np.seterr(all="raise")

from scipy.special import gammaln

from gensim import matutils
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.word2vec import LineSentence
from itertools import combinations
from pathlib import Path
from tqdm.auto import tqdm


def perplexity(N, n_kw, n_dk, alpha, beta):
    likelihood = polyad(n_dk, alpha) + polyaw(n_kw, beta)
    return np.exp(-likelihood/N)


def polyad(n_dk, alpha):
    N = n_dk.shape[0]
    K = n_dk.shape[1]
    n_d = np.sum(n_dk, 1)
    likelihood = np.sum(gammaln(K * alpha) - gammaln(K * alpha + n_d))
    for n in range(N):
        likelihood += np.sum(gammaln(n_dk[n, :] + alpha) - gammaln(alpha))
    return likelihood


def polyaw(n_kw, beta):
    K = n_kw.shape[0]
    V = n_kw.shape[1]
    n_k = np.sum(n_kw, 1)
    likelihood = np.sum(gammaln(V * beta) - gammaln(V * beta + n_k))
    for k in range(K):
        likelihood += np.sum(gammaln(n_kw[k, :] + beta) - gammaln(beta))
    return likelihood


class Progress:
    
    def __init__(self, n_iter):
        self.pbar = tqdm(total=n_iter)

    def update(self, ppl, n=1):
        self.pbar.update(n)
        self.pbar.set_postfix(ppl=f"{ppl:.3f}")        


def norm(x):
    return x/np.sum(x)


def cnorm(M): # column-wise normalize matrix
    s = np.sum(M,0)
    return np.dot(M,np.diag(1.0/s))


def rnorm(M): # row-wise normalize matrix
    return np.array([m/np.sum(m) for m in M])


def document_generator(path):
    doc = []
    for line in open(path, 'r'):
        if re.match(r'^[ \s\u3000]*$', line):
            if len(doc) > 0:
                yield doc
                doc = []
        else:
            words = line.strip().split()
            doc.extend(words)
    if len(doc) > 0:
        yield doc

def detect_input(input, doc_break=False):
    if isinstance(input, str):
        # Path like string input
        if re.fullmatch(r'.{,20}(/[^/]{,20})+', input):
            corpus = detect_input(Path(input))
        else:
            corpus = text2list(input, doc_break)
    elif isinstance(input, list):
        corpus = input
    elif isinstance(input, pathlib.PosixPath):
        if doc_break:
            corpus = document_generator(input)
        else:
            corpus = LineSentence(input)
    else:
        raise ValueError('Invalid Input')

    return corpus


def text2list(text, docbreak=False):
    if docbreak is False:
        return [line.split() for line in text.strip().split('\n')]

    docs, doc = [], []
    for line in text.strip().split('\n'):
        if re.match(r'^[ \s\u3000]*$', line):
            if len(doc) > 0:
                docs.append(doc)
                doc = []
        else:
            words = line.strip().split()
            doc.extend(words)
    if len(doc) > 0:
        docs.append(doc)
    return docs


def read_corpus(corpus):
    D, vocab, word2id = [], [], {}
    for doc in corpus:
        id_doc = []
        for word in doc:
            if word not in word2id:
                word2id[word] = len(vocab)
                vocab.append(word)
            id_doc.append(word2id[word])
        D.append(np.array(id_doc, dtype=np.int32))
    vocab = np.array(vocab, dtype=np.unicode_)
    return D, vocab, word2id


def assign_random_topic(D, K):
    return [np.random.randint(K, size=len(d)) for d in D]


def draw_phi(n_kw, beta):
    K, V = n_kw.shape
    phi = np.empty((V, K), dtype=np.float32)
    n_k = np.sum(n_kw, 1)
    for v in range(V):
        for k in range(K):
            phi[v, k] = (n_kw[k, v] + beta) / (n_k[k] + V * beta)
    return phi


def draw_theta(n_dk, alpha):
    D, K = n_dk.shape
    theta = np.empty((D, K), dtype=np.float32)
    n_d = np.sum(n_dk, 1)
    for d in range(D):
        for k in range(K):
            theta[d, k] = (n_dk[d, k] + alpha) / (n_d[d] + K * alpha)
    return theta


def get_topics(n_kw, vocab, beta, topn=10):
    topics = []
    K, V = n_kw.shape
    n_k = np.sum(n_kw, 1)
    for k in range(K):
        topn_indices = matutils.argsort(n_kw[k, :], topn=topn, reverse=True)
        word_probs = [(vocab[w], (n_kw[k, w] + beta) / (n_k[k] + V * beta)) for w in topn_indices]
        topics.append(word_probs)
    return topics


def get_coherence_model(W, n_kw, top_words, coherence_model, test_texts=None, corpus=None, coo_matrix=None, coo_word2id=None, wv=None, verbose=False):

    if coo_matrix is not None:
        logging.info("Initializing PMI Coherence Model...")
        model = PMICoherence(coo_matrix, coo_word2id, W, n_kw, topn=top_words)
    elif wv is not None:
        logging.info("Initialize Word Embedding Coherence Model...")
        model = EmbeddingCoherence(wv, W, n_kw, topn=top_words)
    else:
        logging.info(f"Initializing {coherence_model} Coherence Model...")
        dictionary = Dictionary.from_documents(corpus)
        if test_texts is not None:
            model = GensimCoherenceModel(coherence_model, test_texts, None, dictionary, W, n_kw, topn=top_words, verbose=verbose)
        else:
            bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
            model = GensimCoherenceModel(coherence_model, None, bow_corpus, dictionary, W, n_kw, topn=top_words, verbose=verbose)
    return model


class GensimCoherenceModel():

    def __init__(self, model, texts, corpus, dictionary, W, n_kw, topn=20, verbose=False):
        self.model = model
        self.texts = texts
        self.corpus = corpus
        self.dictionary = dictionary
        self.W = W
        self.n_kw = n_kw
        self.topn = topn
        self.K = len(n_kw)
        self.verose = verbose

    def get_topics(self):
        topics = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            topics.append([self.W[w] for w in topn_indices])
        return topics

    def score(self):
        topics = self.get_topics()
        if self.model == 'u_mass':
            cm = CoherenceModel(topics=topics,
                                corpus=self.corpus, dictionary=self.dictionary, coherence=self.model)
        else:
            cm = CoherenceModel(topics=topics,
                                texts=self.texts, dictionary=self.dictionary, coherence=self.model)
        if self.verose:
            coherences = cm.get_coherence_per_topic()
            for index, topic in enumerate(topics):
                print(str(index) + ':' + str(coherences[index]) + ':' + ','.join(topic))
        return cm.get_coherence()


class EmbeddingCoherence():

    def __init__(self, wv, W, n_kw, topn=20):
        self.wv = wv
        self.W = W
        self.n_kw = n_kw
        self.topn = topn
        self.K = len(n_kw)

    def score(self):
        scores = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            for x, y in combinations(topn_indices, 2):
                w_x, w_y = self.W[x], self.W[y]
                if w_x in self.wv and w_y in self.wv:
                    scores.append(self.wv.similarity(w_x, w_y))
        return np.mean(scores)


class PMICoherence():

    def __init__(self, M, word2id, W, n_kw, eps=1e-08, topn=20):
        self.M = M
        self.M.setdiag(0)
        self.word2id = word2id
        self.W = W
        self.n_kw = n_kw
        self.eps = eps
        self.topn = topn
        self.K = len(n_kw)
        self.N = np.sum(M)

        V = len(W)
        self.n_w = np.zeros((V), dtype=np.int32)
        for i in tqdm(range(V)):
            if W[i] in word2id:
                self.n_w[i] = self.M[:, word2id[W[i]]].sum()
            else:
                self.n_w[i] = 0

    def pmi(self, x, y, w_x, w_y):
        ix = self.word2id[w_x]
        iy = self.word2id[w_y]
        X = self.n_w[x]
        Y = self.n_w[y]
        XY = self.M[ix, iy]
        if XY == 0 or X == 0 or Y == 0:
            pmi = 0
        else:
            # pmi = np.log2(XY*N/(X*Y+self.eps))/(-np.log(XY/self.N) + self.eps)
            p_xy = XY/self.N
            p_x = X/self.N
            p_y = Y/self.N
            pmi = np.log2(p_xy/(p_x*p_y+self.eps))/(-np.log(p_xy) + self.eps)
        return pmi

    def score(self):
        scores = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            for x, y in combinations(topn_indices, 2):
                w_x, w_y = self.W[x], self.W[y]
                if w_x in self.word2id and w_y in self.word2id:
                    scores.append(self.pmi(x, y, w_x, w_y))
        return np.mean(scores)


def jlh_score(n_kw, beta, vocab, topn=10):
    topics = []
    K, V = n_kw.shape
    n_k = np.sum(n_kw, 1)
    n = np.sum(n_kw)
    jlh_scores = np.zeros((K, V), dtype=np.float32)
    for k in range(K):
        for w in range(V):
            glo = (n_kw[k, w] + V)/(n + V * beta)
            loc = (n_kw[k, w] + V)/(n_k[k] + V * beta)
            jlh_scores[k, w] = (glo-loc) * (glo/loc)
        topn_words = matutils.argsort(jlh_scores[k], topn=topn, reverse=True)
        topics.append([(vocab[w], float(jlh_scores[k, w])) for w in topn_words])
    return topics