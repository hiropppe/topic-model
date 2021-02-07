import logging
import math
import numpy as np
import pandas as pd
import pathlib
import scipy as sci
import tempfile

from pathlib import Path
from tqdm import tqdm

from gensim.corpora import Dictionary

from sklearn.feature_extraction.text import CountVectorizer

from . import sppmi_svd_c


def main(path):
    df = pd.read_csv(path)
    df.dropna(subset=['tokens'], inplace=True)

    max_features = 10000
    threshold = 1
    shift = 10
    has_abs_dis = False
    has_cds = False
    eps = 1e-8

    cv = CountVectorizer(binary=True, lowercase=True, max_features=max_features)
    doc2token = cv.fit_transform((tokens.strip() for tokens in tqdm(df['tokens'].values)))
    doc2token = doc2token.astype(np.float32)

    token2id = cv.vocabulary_
    vocab = cv.get_feature_names()

    C = np.dot(doc2token.T, doc2token).toarray()

    if threshold:
        C = threshold_cooccur(C, threshold)

    M = sppmi_svd_c.sppmi(C, shift, has_abs_dis, has_cds, eps)

    print(M)


def threshold_cooccur(C, threshold):
    """ truncate cooccur matrix by threshold value
    c = c if c > threshold else 0
    :return: fixed cooccur matrix C
    """
    C = np.where(C > threshold, C, 0)

    return C


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
