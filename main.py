# /usr/bin/env python
import click
import logging

from gensim.models import KeyedVectors, Word2Vec

from model.lda import train as clda
from model.pylda import train as pylda
from model.pltm import train as pltm
from model.ctm import train as ctm
from model.nctm import train as nctm


@click.command()
@click.argument("corpus")
@click.option("--model", "-m", default="lda", type=click.Choice(["lda", "pltm", "ctm", "nctm"]))
@click.option("--k", "-k", default=20)
@click.option("--alpha", "-a", default=0.1)
@click.option("--beta", "-b", default=(0.01,), type=float, multiple=True)
@click.option("--gamma", "-g", default=0.01)
@click.option("--eta", "-e", default=1.0)
@click.option("--embedding", default=None)
@click.option("--n_iter", "-i", default=1000)
@click.option("--report_every", default=100)
@click.option("--prefix", default=None)
@click.option("--output_dir", default=".")
@click.option("--py", is_flag=True,  default=False)
def main(corpus, model, k, alpha,  beta, gamma, eta, embedding, n_iter, report_every, prefix, output_dir, py):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if embedding:
        logging.info("Embedding for topic coherence evaluation: {:s}".format(embedding))
        if embedding.endswith(".model"):
            wv = Word2Vec.load(embedding).wv
        else:
            wv = KeyedVectors.load_word2vec_format(embedding)

    if prefix is None:
        prefix = model

    if model == "lda":
        if py:
            lda = pylda
        else:
            lda = clda
        beta = beta[0]
        lda(corpus, k, alpha, beta, wv, n_iter, report_every=report_every,
            prefix=prefix, output_dir=output_dir)
    elif model == "pltm":
        pltm(corpus, k, alpha, beta, n_iter, report_every=report_every)
    elif model == "ctm":
        beta = beta[0]
        ctm(corpus, k, alpha, beta, gamma, n_iter, report_every=report_every)
    elif model == "nctm":
        beta = beta[0]
        nctm(corpus, k, alpha, beta, gamma, eta, wv, n_iter,
             report_every=report_every, prefix=prefix, output_dir=output_dir)
    else:
        raise ValueError("model out of bounds. " + model)


if __name__ == '__main__':
    main()
