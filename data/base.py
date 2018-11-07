import numpy as np
from models.base_transformer import Transformer

class Dataset:

    def __init__(self, loader=None):
        if loader is None: return
        d = loader()
        self._data = np.asarray(d['data'])
        self._labels = np.asarray(d['labels'], dtype=float)
        self._train_idx = np.asarray(d['train_idx'], dtype=int)
        self._valid_idx = np.asarray(d['valid_idx'], dtype=int)
        self._test_idx = np.asarray(d['test_idx'], dtype=int)

    def transform(self, transformer, force_obj=False):
        assert isinstance(transformer, Transformer)
        new_dataset = Dataset()
        if force_obj:
            data = np.empty((self.size,), dtype=object)
            data[:] = transformer.transform(self._data)
            new_dataset._data = data
        else:
            new_dataset._data = np.asarray(transformer.transform(self._data))
        new_dataset._train_idx = self._train_idx
        new_dataset._valid_idx = self._valid_idx
        new_dataset._test_idx = self._test_idx
        new_dataset._labels = self._labels
        return new_dataset

    def replace(self, data, transformer=None):
        if transformer is not None:
            assert isinstance(transformer, Transformer)
            data = transformer.transform(data)
        assert len(data) == len(self._data)
        new_dataset = Dataset()
        new_dataset._data = np.asarray(data)
        new_dataset._train_idx = self._train_idx
        new_dataset._valid_idx = self._valid_idx
        new_dataset._test_idx = self._test_idx
        new_dataset._labels = self._labels
        return new_dataset

    def shuffle(self):
        np.random.shuffle(self._train_idx)

    @property
    def all(self):
        return self._data

    @property
    def all_labels(self):
        return self._labels

    @property
    def size(self):
        return self._data.size

    @property
    def train(self):
        return self._data[self._train_idx]

    @property
    def train_labels(self):
        return self._labels[self._train_idx]

    @property
    def train_size(self):
        return self._train_idx.size

    @property
    def valid(self):
        return self._data[self._valid_idx]

    @property
    def valid_labels(self):
        return self._labels[self._valid_idx]

    @property
    def valid_size(self):
        return self._valid_idx.size

    @property
    def test(self):
        return self._data[self._test_idx]

    @property
    def test_labels(self):
        return self._labels[self._test_idx]

    @property
    def test_size(self):
        return self._test_idx.size

    def __getitem__(self, key):
        return self._data[key], self._labels[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for item in zip(self._data, self._labels):
            yield item
