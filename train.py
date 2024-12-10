#!/usr/bin/env python3
import argparse
import logging
import os
import pickle

from pathlib import Path

from cytm.lda import LDA
from cytm.pltm import PLTM
from cytm.ctm import CTM
from cytm.nctm import NCTM
from cytm.atm import ATM
from cytm.ldab import LDAb

from os.path import expanduser


def lda(args):
    model = LDA(args.Corpus,
                K=args.n_topics,
                alpha=args.alpha,
                beta=args.beta,
                n_iter=args.n_iter,
                report_every=args.report_every)
    return model


def pltm(args):
    model = PLTM(args.Corpus,
                 args.SideInformation,
                 K=args.n_topics,
                 alpha=args.alpha,
                 beta=args.beta,
                 n_iter=args.n_iter)
    return model


def ctm(args):
    model = CTM(args.Corpus,
                args.SideInformation,
                K=args.n_topics,
                alpha=args.alpha,
                beta=args.beta,
                n_iter=args.n_iter)
    return model


def nctm(args):
    model = NCTM(args.Corpus,
                 args.SideInformation,
                 K=args.n_topics,
                 alpha=args.alpha,
                 beta=args.beta,
                 eta=args.eta,
                 n_iter=args.n_iter)
    return model


def atm(args):
    model = ATM(args.Corpus,
                args.Author,
                K=args.n_topics,
                alpha=args.alpha,
                beta=args.beta,
                n_iter=args.n_iter)
    return model


def ldab(args):
    model = LDAb(args.Corpus,
                 K=args.n_topics,
                 alpha=args.alpha,
                 beta=args.beta,
                 eta=args.eta,
                 a0=args.a0,
                 a1=args.a1,
                 n_iter=args.n_iter)
    return model


def add_common_arguments(parser):
    parser.add_argument('Corpus')
    parser.add_argument('--alpha', '-a', default=0.1)
    parser.add_argument('--beta', '-b', default=0.01)
    parser.add_argument('--n_topics', '-K', type=int, default=20)
    parser.add_argument('--n_iter', '-i', type=int, default=100)
    parser.add_argument('--report_every', '-r', type=int, default=10)
    parser.add_argument('--prefix')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)


def main():
    parser = argparse.ArgumentParser(description='Learning Implementation of Statistical Topic Models.')

    subparsers = parser.add_subparsers()
    # LDA
    lda_parser = subparsers.add_parser('lda', help="Latent Dirichlet Allocation")
    add_common_arguments(lda_parser)
    lda_parser.set_defaults(handler=lda)
    # PLTM
    pltm_parser = subparsers.add_parser('pltm', help='Polylingual Topic Model')
    add_common_arguments(pltm_parser)
    pltm_parser.add_argument('SideInformation')
    pltm_parser.set_defaults(handler=pltm)
    # CTM
    ctm_parser = subparsers.add_parser('ctm', help='Correcpondence Topic Model')
    add_common_arguments(ctm_parser)
    ctm_parser.add_argument('SideInformation')
    ctm_parser.add_argument('--gamma', '-g', default=0.01)
    ctm_parser.set_defaults(handler=ctm)
    # NCTM
    nctm_parser = subparsers.add_parser('nctm', help='Noisy Correcpondence Topic Model')
    add_common_arguments(nctm_parser)
    nctm_parser.add_argument('SideInformation')
    nctm_parser.add_argument('--gamma', '-g', default=0.01)
    nctm_parser.add_argument('--eta', '-e', default=1.0)
    nctm_parser.set_defaults(handler=nctm)
    # ATM
    atm_parser = subparsers.add_parser('atm', help='Author Topic Model')
    add_common_arguments(atm_parser)
    atm_parser.add_argument('Author')
    atm_parser.set_defaults(handler=atm)
    # LDAb
    ldab_parser = subparsers.add_parser('ldab', help='LDA with a background distribution')
    add_common_arguments(ldab_parser)
    ldab_parser.add_argument('--eta', '-e', default=0.005)
    ldab_parser.add_argument('--a0', '-a0', default=2.0)
    ldab_parser.add_argument('--a1', '-a1', default=1.0)
    ldab_parser.set_defaults(handler=ldab)

    args = parser.parse_args()

    if hasattr(args, 'handler'):
        if args.verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        args.Corpus = Path(expanduser(args.Corpus))
        args.output_dir = Path(expanduser(args.output_dir))

        if hasattr(args, 'SideInformation'):
            args.SideInformation = Path(expanduser(args.SideInformation))

        if hasattr(args, 'Author'):
            args.Author = Path(expanduser(args.Author))
        
        if not args.prefix:
            args.prefix = args.handler.__name__

        model = args.handler(args)

        os.makedirs(args.output_dir, exist_ok=True)
        model_path = args.output_dir / f'{args.prefix}.model'
        with open(model_path, mode='wb') as f:
            pickle.dump(model, f)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
