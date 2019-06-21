# /usr/bin/env python
import click
import logging

from model.lda import train as lda
from model.pylda import train as pylda
from model.pltm import train as pltm


@click.command()
@click.argument("corpus")
@click.option("--model", "-m", default="lda", type=click.Choice(["lda", "pltm"]))
@click.option("--k", "-k", default=20)
@click.option("--alpha", "-a", default=0.1)
@click.option("--beta", "-b", default=(0.01,), type=float, multiple=True)
@click.option("--n_iter", "-i", default=1000)
@click.option("--py", is_flag=True,  default=False)
def main(corpus, model, k, alpha,  beta, n_iter, py):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if model == "lda":
        if py:
            train = pylda
        else:
            train = lda
    elif model == "pltm":
        train = pltm
    else:
        raise ValueError("model out of bounds. " + model)

    if model != "pltm":
        beta = beta[0]

    train(corpus, k, alpha, beta, n_iter)


if __name__ == '__main__':
    main()
