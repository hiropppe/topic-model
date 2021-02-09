import numpy as np
import pandas as pd
import tempfile

from pathlib import Path
from tqdm import tqdm

from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer

from . import sppmi_svd_c


def main(path):
    global M, Y, D, W, R

    df = pd.read_csv(path)
    df.dropna(subset=['tokens'], inplace=True)

    max_features = 10000
    threshold = 1
    shift = 10
    has_abs_dis = False
    has_cds = False
    eps = 1e-8
    K = 100

    cv = CountVectorizer(binary=True, lowercase=True, max_features=max_features)
    doc2token = cv.fit_transform((tokens.strip() for tokens in tqdm(df['tokens'].values)))
    doc2token = doc2token.astype(np.float32)

    #M = np.dot(doc2token.T, doc2token).toarray() # for wv
    M = doc2token.toarray()

    if threshold:
        M = threshold_cooccur(M, threshold)

    Y = sppmi_svd_c.sppmi(M, shift, has_abs_dis, has_cds, eps)

    U, S, V = svds(Y, k=K)

    #word_vec = np.dot(U, np.sqrt(np.diag(S)))
    #np.save(tmpdir / 'wv.npy', word_vec[:, :K])

    D = U
    W = V.T
    R = np.linalg.solve(np.dot(W.T, W), W.T)

    tmpdir = Path(tempfile.mkdtemp())
    np.save(tmpdir / 'Y.npy', Y)
    np.save(tmpdir / 'D.npy', D)
    np.save(tmpdir / 'W.npy', W)
    np.save(tmpdir / 'R.npy', R)


def threshold_cooccur(M, threshold):
    """ truncate cooccur matrix by threshold value
    m = m if m > threshold else 0
    :return: fixed cooccur matrix C
    """
    M = np.where(M > threshold, M, 0)

    return M


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
