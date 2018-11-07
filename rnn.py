#!/usr/bin/env python3

import pickle as pk
import argparse

from gensim.corpora import Dictionary
from models import RNNModel

if __name__ == '__main__':

    # Read command-line arguments
    parser = argparse.ArgumentParser(description="Generic NN model.")
    parser.add_argument('datafile', metavar='PATH',
                        help="Pickled dataset file (assumes integer sequences)")
    parser.add_argument('-d', '--dictionary', metavar='PATH', default='dict.pk',
                        help="Pickled dictionary file (Gensim)")
    parser.add_argument('-e', '--epochs', type=int, metavar='N', default=5,
                        help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, metavar='N',
                        default=32, help="Batch size used in training.")
    parser.add_argument('-l', '--load', metavar='FILE',
                        help="Load model from file.")
    parser.add_argument('-s', '--save', metavar='FILE',
                        help="Save model to file.")
    parser.add_argument('-v', '--vector_size',
                        metavar='SIZE', type=int, default=0,
                        help="Size of input vectors (if sequence of vectors)")

    args = parser.parse_args()

    dictionary = Dictionary.load(args.dictionary)

    # Input dataset
    data = pk.load(open(args.datafile, 'rb'))

    model = RNNModel(vocab_size=len(dictionary), load=args.load,
                     vector_size=args.vector_size)

    # Fit and test expect a Dataset object (they use the proper subset)
    model.fit(data, epochs=args.epochs, batch_size=args.batch_size)
    model.test(data, batch_size=args.batch_size)

    if args.save:
        model.save(args.save)
