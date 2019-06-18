#/usr/bin/env python
import click
import logging

from model.lda import LDA
from model.clda import LDA as cLDA


@click.command()
@click.argument("corpus")
@click.option("--k", "-k", default=20)
@click.option("--alpha", "-a", default=0.1)
@click.option("--beta", "-b", default=0.01)
@click.option("--n_iter", "-n", default=50)
@click.option("--py", is_flag=True,  default=False)
def main(corpus, k, alpha,  beta, n_iter, py):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if py:
        lda = LDA(corpus, k, alpha, beta)
    else:
        lda = cLDA(corpus, k, alpha, beta)
    lda.inference(n_iter)
    lda.save('test')


if __name__ == '__main__':
    main()
