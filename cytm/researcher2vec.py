import numpy as np

from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .sppmi_c import sppmis

eps = 1e-8


class Researcher2Vec():

    def __init__(self,
                 corpus,
                 user2doc,
                 K = 100,
                 shift=1,
                 max_features=None):
        self.K = K
        self.shift = shift
        self.max_features = max_features
        
        self.cv = CountVectorizer(binary=True, lowercase=True, max_features=self.max_features)
        Co = self.cv.fit_transform((tokens.strip() for tokens in tqdm(corpus))).astype(np.float32)
        Y = sppmis(Co, self.shift, eps)
        U, S, V = svds(Y, k=self.K)
        W = V.T
        self.document_vectors = np.dot(U, np.sqrt(np.diag(S)))
        self.R = np.linalg.solve(np.dot(W.T, W), W.T)

        self.users = list(user2doc.keys())
        self.user2id, user_vectors = {}, []
        for i, (user, indices) in enumerate(user2doc.items()):
            user_vectors.append(self.document_vectors[indices].mean(axis=0))
            self.user2id[user] = i
        self.user_vectors = np.array(user_vectors)

    def __dict__(self, user):
        return self.__user_vector(user)

    def encode(self, words):
        y = self.cv.transform([words]).toarray()[0]
        d = np.dot(self.R, y)
        return d

    def similars(self, user, topn=None):
        if not topn:
            topn = len(self.users)
        me = self.__user_vector(user)
        all_users = [(u[1], self.cosine_similarity(me, self.user_vectors[u[0]])) for u in enumerate(self.users)] 
        return sorted(all_users, key=lambda u: u[1], reverse=True)[:topn] 

    def search_by_words(self, words, topn=None):
        if not topn:
            topn = len(self.users)
        d = self.encode(words)
        all_users = [(u[1], self.cosine_similarity(d, self.user_vectors[u[0]])) for u in enumerate(self.users)] 
        return sorted(all_users, key=lambda u: u[1], reverse=True)[:topn] 

    def cosine_similarity(self, v1 ,v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def __user_vector(self, user):
        return self.user_vectors[self.user2id[user]]

