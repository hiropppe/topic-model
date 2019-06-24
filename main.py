# /usr/bin/env python
import click
import logging

from model.lda import train as clda
from model.pylda import train as pylda
from model.pltm import train as pltm
from model.ctm import train as ctm


@click.command()
@click.argument("corpus")
@click.option("--model", "-m", default="lda", type=click.Choice(["lda", "pltm", "ctm"]))
@click.option("--k", "-k", default=20)
@click.option("--alpha", "-a", default=0.1)
@click.option("--beta", "-b", default=(0.01,), type=float, multiple=True)
@click.option("--gamma", "-g", default=0.01)
@click.option("--n_iter", "-i", default=1000)
@click.option("--py", is_flag=True,  default=False)
def main(corpus, model, k, alpha,  beta, gamma, n_iter, py):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if model == "lda":
        if py:
            lda = pylda
        else:
            lda = clda
        beta = beta[0]
        lda(corpus, k, alpha, beta, n_iter)
    elif model == "pltm":
        pltm(corpus, k, alpha, beta, n_iter)
    elif model == "ctm":
        beta = beta[0]
        ctm(corpus, k, alpha, beta, gamma, n_iter)
    else:
        raise ValueError("model out of bounds. " + model)


if __name__ == '__main__':
    main()
