#!/usr/local/bin/python
#
#    lda.py
#    Latent Dirichlet allocation with a Gibbs sampling on Cython.
#    The Cython core is based on the code of Ryan P. Adams (Harvard at that time).
#    $Id: lda.py,v 1.19 2023/06/20 11:13:07 daichi Exp $
#
import argparse
import sys
import gzip
import pickle
import numpy as np
from numpy.random import rand, randint
from numpy.random import beta as betarand
from numpy.random import gamma as gamrand
from scipy.special import gammaln
from numpy import exp,log

sys.path.append("./lda.py-0.3/")
import ldac
import fmatrix
import loggings
from loggings import elprintf


def usage ():
    print ('usage: % lda.py OPTIONS train model [lexicon]')
    print ('OPTIONS')
    print (' -K topics  number of topics in LDA')
    print (' -N iters   number of Gibbs iterations (default 1)')
    print (' -a alpha   Dirichlet hyperparameter on topics (default auto)')
    print (' -b beta    Dirichlet hyperparameter on words (default auto)')
    print (' -h         displays this help')
    print ('$Id: lda.py,v 1.19 2023/06/20 11:13:07 daichi Exp $')
    sys.exit (0)

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("model")
    parser.add_argument("-K", type=int, default=10)
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-a", type=float)
    parser.add_argument("-b", type=float)
    args = parser.parse_args()

    train = args.train
    model = args.model
    #lexicon = args[2] if len(args) > 2 else None
    lexicon = None
    K = args.K
    iters = args.N 
    alpha = np.ones (K, dtype=float) * (args.a if args.a else 50 / K)
    alpha_auto = False if args.a else True
    beta  = args.b if args.b else 0.01
    beta_auto = False if args.b else True

    eprint('LDA: K = %d, iters = %d, alpha = %g, beta = %g' \
            % (K,iters,alpha[0],beta))
    loggings.start (sys.argv, model)
          
    eprintf('loading data.. ')
    W,Z = ldaload (train, K)
    D   = len(W)
    V   = nlex(W)
    N   = datalen(W)
    NW  = np.zeros ((V,K), dtype=np.int32)
    ND  = np.zeros ((D,K), dtype=np.int32)
    NZ  = np.zeros (K, dtype=np.int32)
    seen = ldaseen (train)
    eprint ('documents = %d, lexicon = %d, nwords = %d' % (D,V,N))

    # initialize
    eprint('initializing..')
    ldac.init (W, Z, NW, ND, NZ)

    for iter in range(iters):
        ppl = ldappl (W, Z, NW, ND, alpha, beta)
        elprintf(('Gibbs iteration [%2d/%2d] PPL = %.2f (joint=%.2f) ' +
                  'alpha=%.3f beta=%.3f') \
                 % (iter+1, iters, ppl[0], ppl[1], np.mean(alpha), beta))
        ldac.gibbs (W, Z, NW, ND, NZ, alpha, beta)
        if alpha_auto:
            alpha = draw_alpha (ND, alpha)
        if beta_auto:
            beta = draw_beta (NW, beta)
    eprintf('\n')
    
    save (model, lexicon, seen, W, Z, NW, ND, alpha, beta)
    loggings.finish ()

#
#   inference functions.
#

def draw_alpha (ND, alpha):
    a = 1; b = 1
    D,K = ND.shape
    nw = np.sum (ND, 1)
    s = np.zeros (K, dtype=float)
    theta = np.zeros (D, dtype=float)
    alpha0 = np.sum (alpha)
    for d in range(D):
        theta[d] = betarand (alpha0, nw[d])
    for d in range(D):
        for k in range(K):
            if ND[d][k] > 0:
                s[k] += draw_ykv (ND[d][k], alpha[k])
    for k in range(K):
        alpha[k] = gamrand (a + s[k], 1 / (b - np.sum (log(theta))))
    return alpha

def draw_beta (NW, beta):
    a = 1; b = 1
    V,K = NW.shape
    theta = np.zeros (K, dtype=float)
    nk = np.sum (NW, 0)
    for k in range(K):
        theta[k] = betarand (V * beta, nk[k])
    s = 0
    for v in range(V):
        for k in range (K):
            if NW[v][k] > 0:
                s += draw_ykv (NW[v][k], beta)
    beta = gamrand (a + s, 1 / (b - V * np.sum (log(theta))))
    return beta

def draw_ykv (m, eta):
    s = 0
    for i in range(m):
        s += bernoulli (eta / (eta + i))
    return s

def ldappl (W, Z, NW, ND, alpha, beta):
    L = datalen(W)
    likd = polyad (ND, alpha)
    likw = polyaw (NW, beta)
    return np.exp (- likw / L), np.exp (- (likw + likd) / L)

def polyad (ND, alpha):
    D = ND.shape[0]
    K = ND.shape[1]
    nd = np.sum (ND,1)
    lik = np.sum (gammaln (np.sum(alpha)) - gammaln (np.sum(alpha) + nd))
    for n in range(D):
        lik += np.sum (gammaln (ND[n,:] + alpha) - gammaln (alpha))
    return lik

def polyaw (NW, beta):
    V = NW.shape[0]
    K = NW.shape[1]
    nw = np.sum (NW,0)
    lik = np.sum (gammaln (V * beta) - gammaln (V * beta + nw))
    for k in range(K):
        lik += np.sum (gammaln (NW[:,k] + beta) - gammaln (beta))
    return lik

#
#   I/O functions.
#

def save (file, lexicon, seen, W, Z, NW, ND, alpha, beta):
    eprint ('saving model to %s .. ' % file)
    model = {}
    model['alpha'] = alpha
    model['nw'] = NW
    model['nd'] = ND
    model['beta']  = cnormalize (NW + beta)
    model['theta'] = rnormalize (ND + alpha)
    model['seen']  = seen
    if lexicon:
        eprint ('including dictionary.')
        dic = readdic (lexicon)
        model['lexicon'] = dic
    # save to file
    with gzip.open (file, 'wb') as gf:
        pickle.dump (model, gf, 2)
    eprint ('done.')

def readdic (file):
    dic = {}
    with open (file, 'r') as fh:
        for line in fh:
            id,word = line.rstrip('\n').split('\t')
            dic[word] = int(id)
    return dic
    
def ldaload (file, K):
    words = fmatrix.plain (file)
    topics = randtopic (words, K)
    return int32(words), int32(topics)
    
def ldaseen (file):
    seen = {}
    data = fmatrix.parse (file)
    for doc in data:
        for v in doc.id:
            seen[v] = True
    return seen

#
#   utility functions.
#

def datalen (W):
    return sum (map (lambda x: x.shape[0], W))

def randtopic (words, K):
    topics = []
    for word in words:
        topic = randint (K, size=len(word))
        topics.append (topic)
    return topics

def nlex (words):
    v = 0
    for word in words:
        if (len(word) > 0) and (max(word) > v):
            v = max(word)
    return v + 1

def int32 (words):
    data = []
    for word in words:
        data.append (np.array(word, dtype=np.int32))
    return data

def datalen (W):
    n = 0
    for w in W:
        n += len(w)
    return n

def eprint (s):
    sys.stderr.write (s + '\n')
    sys.stderr.flush ()

def eprintf (s):
    sys.stderr.write (s)
    sys.stderr.flush ()

def bernoulli (p):
    if rand() < p:
        return 1
    else:
        return 0

def cnormalize (M): # column-wise normalize matrix
    s = np.sum(M,0)
    return np.dot(M,np.diag(1.0/s))

def rnormalize (M): # row-wise normalize matrix
    return np.array ([m / np.sum(m) for m in M])
    
if __name__ == "__main__":
    main ()
