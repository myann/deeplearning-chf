from .base_model import Model
from .base_transformer import Transformer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np

class Doc2VecModel(Model, Transformer):

    def __init__(self, corpus=None, **kwargs):
        self._m = Doc2Vec(documents=None, **kwargs)
        if corpus is not None:
            self._m.build_vocab(corpus)

    def fit(self, corpus, iters=10):
        corpus = np.copy(corpus)
        N = len(corpus)
        for _ in range(iters):
            self._m.train(corpus, total_examples=N)
            np.random.shuffle(corpus)

    def transform(self, corpus):
        return [self._m.infer_vector(doc) for doc in corpus]

    @property
    def inst(self):
        return self._m

class DocTagger(Transformer):

    def transform(self, corpus):
        return [TaggedDocument(words, [i]) for i, words in enumerate(corpus)]

class Word2VecTransformer(Transformer):

    def __init__(self, model):
        self._m = model._m

    def transform(self, corpus):
        return [np.asarray([self._m[word] for word in sentence])
                for sentence in corpus]
