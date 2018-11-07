#!/usr/bin/env python3

import pickle as pk
import numpy as np
import argparse
import scipy
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from sklearn.manifold import TSNE

def transform(data, N):
    new_data = scipy.sparse.lil_matrix((len(data), N), dtype=np.float32)
    for i, row in enumerate(data):
        idx = [v[0] for v in row]
        val = [v[1] for v in row]
        new_data[i, idx] = val
    return new_data.tocsr()

if __name__ == '__main__':

    # Read command-line arguments
    parser = argparse.ArgumentParser(description="t-SNE model.")
    parser.add_argument('datafile', metavar='PATH',
                        help="Pickled dataset file (assumes sparse vectors)")
    parser.add_argument('-d', '--dictionary', metavar='PATH', default='dict.pk',
                        help="Pickled dictionary file (Gensim)")
    parser.add_argument('-p', '--perplexity', metavar='NUM',
                        default=30, type=int)

    args = parser.parse_args()

    data = pk.load(open(args.datafile, 'rb'))

    dictionary = Dictionary.load(args.dictionary)
    N = len(dictionary)  # vocabulary size

    X = transform(data.all, N)

    tsne = TSNE(method='exact', perplexity=args.perplexity)
    print("Fitting t-SNE...")
    Y = tsne.fit_transform(X)
    print("Kullback-Leibler divergence: {}".format(tsne.kl_divergence_))

    plt.scatter(Y[:, 0], Y[:, 1], s=10, c=data.all_labels)
    plt.show()
