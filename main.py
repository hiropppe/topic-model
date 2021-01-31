# /usr/bin/env python
import click
import logging
import scipy as sci

from gensim.models import KeyedVectors, Word2Vec

from cytm.lda import train as clda
from cytm.pylda import train as pylda
from cytm.pltm import train as pltm
from cytm.ctm import train as ctm
from cytm.nctm import train as nctm
from cytm.atm import train as atm

@click.command()
@click.argument("corpus")
@click.option("--model", "-m", default="lda", type=click.Choice(["lda", "pltm", "ctm", "nctm", "atm"]))
@click.option("--k", "-k", default=20)
@click.option("--alpha", "-a", default=0.1)
@click.option("--beta", "-b", default=(0.01,), type=float, multiple=True)
@click.option("--gamma", "-g", default=0.01)
@click.option("--eta", "-e", default=1.0)
@click.option("--top_words", "-topn", default=20)
@click.option("--coherence_model", "-cm", default="u_mass", type=click.Choice(["u_mass", "c_v", "c_uci", "c_npmi"]))
@click.option("--test_data", default=None)
@click.option("--embedding", default=None)
@click.option("--coo_prefix", default=None)
@click.option("--n_iter", "-i", default=100)
@click.option("--report_every", "-r", default=10)
@click.option("--prefix", default=None)
@click.option("--output_dir", default=".")
@click.option("--py", is_flag=True,  default=False)
@click.option("--verbose", "-v", is_flag=True,  default=False)
def main(corpus,
         model,
         k,
         alpha,
         beta,
         gamma,
         eta,
         top_words,
         coherence_model,
         test_data,
         embedding,
         coo_prefix,
         n_iter,
         report_every,
         prefix,
         output_dir,
         py,
         verbose):
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    wv = load_wv(embedding)

    coo_matrix, coo_word2id, coo_vocab = load_coocurence_matrix(coo_prefix)

    if test_data is not None:
        test_texts = [text[:-1].split() for text in open(test_data)]
    else:
        test_texts = None

    if prefix is None:
        prefix = model

    if model == "lda":
        if py:
            lda = pylda
        else:
            lda = clda
        beta = beta[0]
        lda(corpus,
            k,
            alpha,
            beta,
            top_words=top_words,
            coherence_model=coherence_model,
            test_texts=test_texts,
            wv=wv,
            coo_matrix=coo_matrix,
            coo_word2id=coo_word2id,
            n_iter=n_iter,
            report_every=report_every,
            prefix=prefix,
            output_dir=output_dir,
            verbose=verbose)
    elif model == "atm":
        beta = beta[0]
        atm(corpus,
            k,
            alpha,
            beta,
            top_words=top_words,
            n_iter=n_iter,
            report_every=report_every,
            prefix=prefix,
            output_dir=output_dir,
            verbose=verbose)
    elif model == "pltm":
        pltm(corpus,
             k,
             alpha,
             beta,
             n_iter,
             report_every=report_every)
    elif model == "ctm":
        beta = beta[0]
        ctm(corpus,
            k,
            alpha,
            beta,
            gamma,
            n_iter,
            report_every=report_every)
    elif model == "nctm":
        beta = beta[0]
        nctm(corpus,
             k,
             alpha,
             beta,
             gamma,
             eta,
             wv,
             coo_matrix,
             coo_word2id,
             n_iter,
             report_every=report_every,
             prefix=prefix,
             output_dir=output_dir)
    else:
        raise ValueError("model out of bounds. " + model)


def load_wv(embedding):
    if embedding:
        logging.info("Loading Embedding for topic coherence evaluation: {:s}".format(embedding))
        if embedding.endswith(".model"):
            wv = Word2Vec.load(embedding).wv
        else:
            wv = KeyedVectors.load_word2vec_format(embedding)
        return wv


def load_coocurence_matrix(coo_prefix):
    if coo_prefix:
        logging.info(
            "Loading Co-occurence matrix for topic coherence evaluation: {:s}".format(coo_prefix))
        inp_csc_path = coo_prefix + ".csc"
        inp_token2id_path = coo_prefix + ".token2id"
        inp_vocab_path = coo_prefix + ".vocab"

        with open(inp_csc_path, "rb") as fi:
            coo_matrix = sci.sparse.load_npz(fi)

        coo_word2id = {}
        with open(inp_token2id_path) as fi:
            for line in fi:
                line = line[:-1]
                item = line.split()
                coo_word2id[item[0]] = int(item[1])

        coo_vocab = []
        with open(inp_vocab_path) as fi:
            for line in fi:
                coo_vocab.append(line[:-1])
    else:
        coo_matrix = None
        coo_word2id = None
        coo_vocab = None

    return coo_matrix, coo_word2id, coo_vocab


if __name__ == '__main__':
    main()
