from .base_model import Model
from .base_transformer import Transformer

from gensim.models import TfidfModel
import numpy as np

class TFIDFModel(Model, Transformer):

    def __init__(self, corpus=None, **kwargs):
        self._m = TfidfModel(corpus, **kwargs)

    def fit(self, corpus):
        self._m.initialize(corpus)

    def transform(self, corpus):
        return self._m[corpus]

    @property
    def inst(self):
        return self._m
