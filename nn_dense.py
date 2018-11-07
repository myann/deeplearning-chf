import pickle as pk
import numpy as np
import tensorflow as tf
import argparse

from keras.layers import Input, Dense, Dropout
from keras.models import Model

from gensim.corpora import Dictionary

if __name__ == '__main__':

    # Read command-line arguments
    parser = argparse.ArgumentParser(description='Generic NN model.')
    parser.add_argument('datafile', metavar='PATH',
                        help='Pickled dataset file (assumes dense vectors)')
    parser.add_argument('-e', '--epochs', type=int, metavar='N', default=5,
                        help='Number of epochs to train for')
    parser.add_argument('-b', '--batch_size', type=int, metavar='N',
                        default=32, help="Batch size used in training.")

    args = parser.parse_args()

    # Input dataset
    data = pk.load(open(args.datafile, 'rb'))

    # Model definition
    x = Input(shape=(data.all[0].shape[-1],))
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

    # Train the model!
    model.fit(
        data.train, data.train_labels, batch_size=args.batch_size,
        nb_epoch=args.epochs,
        validation_data=(data.valid, data.valid_labels)
    )

    # Evaluate final accuracy
    metrics = model.evaluate(
        data.test, data.test_labels, batch_size=args.batch_size
    )
    print("\nFinal accuracy: {:.2%}".format(metrics[1]))
