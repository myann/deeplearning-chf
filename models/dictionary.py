from .base_model import Model
from .base_transformer import Transformer

from gensim.corpora import Dictionary as Dict

class Dictionary(Model, Transformer):

    def __init__(self, corpus=None):
        self._d = Dict(corpus)

    def fit(self, corpus):
        self._d.add_documents(corpus)

    def transform(self, corpus):
        return [[self._d.token2id.get(t, -2) + 1 for t in row]
                for row in corpus]

    def to_bow(self, corpus):
        return [self._d.doc2bow(row) for row in corpus]

    @property
    def size(self):
        return len(self._d)

    @property
    def inst(self):
        return self._d
