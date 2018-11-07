from __future__ import print_function
import tensorflow as tf
import threading as th
import numpy as np
from glob import glob
from math import ceil

from keras.layers import Dense, GRU, Embedding
from keras.metrics import binary_accuracy as accuracy
from keras.objectives import binary_crossentropy as crossentropy
from keras.models import Sequential
from keras import backend as K

class RNNModel:

    def __init__(self, qsize=256, vocab_size=10000, vector_size=0, load=None):
        # Create one model for each bucket
        self._model = self._buildModel(qsize, vocab_size, vector_size)
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        if load is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            self._saver.restore(self._sess, load)

    def _buildModel(self, qsize, vsize, vector_size):
        t_shape, t_dtype = (), tf.float32
        if not vector_size:
            x_shape = (None,)
            x_dtype = tf.int32
        else:
            x_shape = (None, vector_size)
            x_dtype = tf.float32
        x_in = tf.placeholder(x_dtype, shape=x_shape)
        t_in = tf.placeholder(t_dtype, shape=t_shape)
        batch_size = tf.placeholder_with_default(tf.constant(64), shape=())

        # Input queue
        q = tf.PaddingFIFOQueue(qsize, (x_dtype, t_dtype),
                                shapes=(x_shape, t_shape))
        enqueue_op = q.enqueue((x_in, t_in))

        # Fetched variables
        x, t = q.dequeue_many(batch_size)

        # Model definition
        model = Sequential()
        if not vector_size:
            model.add(Embedding(output_dim=512,
                                input_dim=vsize+2, mask_zero=True))
            model.add(GRU(32))
        else:
            model.add(GRU(32, input_dim=vector_size))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        y = K.squeeze(model(x), 1)  # shape = (batch_size,) - same as t

        # Metrics
        loss = K.mean(K.binary_crossentropy(y, t))
        acc = accuracy(t, y)

        # Trainer
        train_op = tf.train.AdamOptimizer().minimize(loss)

        return {
            'x': x_in,
            't': t_in,
            'y': y,
            'q': q,
            'acc': acc,
            'loss': loss,
            'train_op': train_op,
            'enqueue_op': enqueue_op,
            'batch_size': batch_size
        }

    def fit(self, data, epochs=10, batch_size=64):

        coord = tf.train.Coordinator()
        stats = {'losses': [], 'accuracies': [], 'count': 0}
        valid_stats = {'accuracies': [], 'losses': [], 'count': 0}
        m = self._model
        sess = self._sess

        # Trainer code
        def train():
            remaining = data.train_size
            # print("Starting trainer ...")

            for i in range(epochs):
                print("Epoch {}".format(i+1))

                while remaining:
                    if coord.should_stop(): return

                    batch_size_ = min(batch_size, remaining)
                    try:
                        # Run a single train pass (eval and backprop)
                        r = sess.run([m['train_op'], m['loss'], m['acc']],
                                     feed_dict={m['batch_size']: batch_size_})
                    except tf.errors.OutOfRangeError:
                        # Queue is closed, and nothing remains
                        return
                    except Exception as e:
                        # Some unexpected error occurred, halt everything
                        coord.request_stop(e)
                        return

                    stats['count'] += batch_size_
                    stats['losses'].append(r[1])
                    stats['accuracies'].append(r[2])
                    remaining -= batch_size_

                    print("@{}/{} - loss: {:.5f}, acc: {:.2%}".format(
                            stats['count'], data.train_size,
                            np.mean(stats['losses']),
                            np.mean(stats['accuracies'])
                          ), end='\r')

                print(''.join([' '] * 80), end='\r')  # clear last line

                valid_stats['count'] = 0
                valid_stats['losses'] = []
                valid_stats['accuracies'] = []
                remaining = data.valid_size
                while remaining:
                    if coord.should_stop(): return

                    batch_size_ = min(batch_size, remaining)
                    try:
                        # Run a single train pass (eval and backprop)
                        r = sess.run([m['loss'], m['acc']],
                                     feed_dict={m['batch_size']: batch_size_})
                    except tf.errors.OutOfRangeError:
                        # Queue is closed, and nothing remains
                        return
                    except Exception as e:
                        # Some unexpected error occurred, halt everything
                        coord.request_stop(e)
                        return
                    valid_stats['count'] += batch_size_
                    valid_stats['losses'].append(r[0])
                    valid_stats['accuracies'].append(r[1])
                    remaining -= batch_size_

                # Print epoch statistics
                if valid_stats['count']:
                    print("Stats: loss {:.5f}, acc {:.2%}, count {} "
                          "- val_loss {:.5f}, val_acc {:.2%}, val_count {}"
                          .format(np.mean(stats['losses']),
                                  np.mean(stats['accuracies']),
                                  stats['count'],
                                  np.mean(valid_stats['losses']),
                                  np.mean(valid_stats['accuracies']),
                                  valid_stats['count']))
                else:  # if there's no validation set
                    print("Stats: loss {:.5f}, acc {:.2%}, count {} "
                          .format(np.mean(stats['losses']),
                                  np.mean(stats['accuracies']),
                                  stats['count']))

                # Reset stats each epoch
                stats['count'] = 0
                stats['losses'] = []
                stats['accuracies'] = []
                remaining = data.train_size

            coord.request_stop()
            # print("Ending trainer...")

        # Create the trainer thread
        trainer = th.Thread(target=train)
        trainer.start()

        try:
            # Start sending out data
            for _ in range(epochs):
                # Send training set
                for x, t in zip(data.train, data.train_labels):
                    if coord.should_stop(): break
                    sess.run(m['enqueue_op'], feed_dict={m['x']: x, m['t']: t})
                # Send validation set
                for x, t in zip(data.valid, data.valid_labels):
                    if coord.should_stop(): break
                    sess.run(m['enqueue_op'], feed_dict={m['x']: x, m['t']: t})
                data.shuffle()  # shuffle at epoch end
        except KeyboardInterrupt:
            print(''.join([' '] * 80), end='\r')  # clear last line
            print("User cancelled.")
            coord.request_stop()
        except tf.errors.CancelledError:
            pass  # trainer closed up the queue, no problem
        except Exception as e:  # Unexpected exception
            coord.request_stop(e)

        coord.join([trainer])

    def test(self, data, batch_size=64):
        coord = tf.train.Coordinator()
        results = {'accuracies': [], 'count': 0}
        m = self._model
        sess = self._sess

        # Evaluator code
        def evaluator():
            remaining = data.test_size
            # print("Starting evaluator...")
            while remaining:
                if coord.should_stop(): break
                batch_size_ = min(batch_size, remaining)
                try:
                    # Run a single train pass (eval and backprop)
                    acc = sess.run(m['acc'],
                                   feed_dict={m['batch_size']: batch_size_})
                except tf.errors.OutOfRangeError:
                    # Queue is closed, but nothing remains
                    break
                except Exception as e:
                    # Some unexpected error occurred, halt everything
                    coord.request_stop(e)
                    break
                results['accuracies'].append(acc)
                results['count'] += batch_size_
                remaining -= batch_size_
            coord.request_stop()
            # print("Ending evaluator...")

        # Create the trainer threads
        evaluator = th.Thread(target=evaluator)
        evaluator.start()

        try:
            for x, t in zip(data.test, data.test_labels):
                if coord.should_stop(): break
                sess.run(m['enqueue_op'], feed_dict={m['x']: x, m['t']: t})
        except KeyboardInterrupt:
            print("User cancelled.")
            coord.request_stop()
        except Exception as e: # Unexpected exception
            coord.request_stop(e)

        coord.join([evaluator])

        print("Accuracy: {} (over {} test samples)"
              .format(np.mean(results['accuracies']), results['count']))

    def save(self, path):
        self._saver.save(self._sess, path)

    def load(self, path):
        self._saver.restore(self._sess, path)
