import pickle as pk
import numpy as np
import tensorflow as tf
import argparse

from keras.layers import Input, Dense
from keras.models import Model

from gensim.corpora import Dictionary

if __name__ == '__main__':

    # Read command-line arguments
    parser = argparse.ArgumentParser(description='Generic NN model.')
    parser.add_argument('datafile', metavar='PATH',
                        help='Pickled dataset file (assumes sparse vectors)')
    parser.add_argument('-d', '--dictionary', metavar='PATH', default='dict.pk',
                        help='Pickled dictionary file (Gensim)')
    parser.add_argument('-e', '--epochs', type=int, metavar='N', default=5,
                        help='Number of epochs to train for')
    parser.add_argument('-q', '--queue_size', type=int, metavar='SIZE',
                        default=512, help='Maximum queue size.')
    parser.add_argument('--size', type=int,
                        help='Override size for alternate length vectors.')

    args = parser.parse_args()

    if args.size:
        N = args.size
    else:
        dictionary = Dictionary.load(args.dictionary)
        N = len(dictionary)  # vocabulary size

    # Model definition
    x = Input(shape=(N,))
    z = Dense(256, activation='relu')(x)
    z = Dense(256, activation='relu')(z)
    z = Dense(256, activation='relu')(z)
    y = Dense(1, activation='sigmoid')(z)

    model = Model(input=x, output=y)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']  # add other metrics
    )

    # Input dataset
    data = pk.load(open(args.datafile, 'rb'))

    # Generates batches of size 1 (converts x to dense vectors)
    def gen(data_x, data_y, shuffle=True):
        while True:
            for x, y in zip(data_x, data_y):
                idx = [v[0] for v in x]
                val = [v[1] for v in x]
                z = np.zeros(N)
                z[idx] = val
                yield np.reshape(z, (1, -1)), np.reshape(y, (1, -1))
            data.shuffle()  # shuffle at epoch end

    # Train the model!
    model.fit_generator(
        gen(data.train, data.train_labels), data.train_size,
        nb_epoch=args.epochs,
        validation_data=gen(data.valid, data.valid_labels, shuffle=False),
        nb_val_samples=data.valid_size,
        max_q_size=args.queue_size
    )

    # Evaluate final accuracy
    metrics = model.evaluate_generator(
        gen(data.test, data.test_labels, shuffle=False), data.test_size,
        max_q_size=args.queue_size
    )
    print("Final accuracy: {:.2%}".format(metrics[1]))
