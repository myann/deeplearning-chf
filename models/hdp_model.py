from .base_model import Model
from .base_transformer import Transformer

from gensim.models import HdpModel
import numpy as np

class HDPModel(Model, Transformer):

    def __init__(self, corpus=None, **kwargs):
        self._m = HdpModel(corpus, **kwargs)

    def fit(self, corpus):
        self._m.update(corpus)

    def transform(self, corpus):
        return self._m[corpus]

    @property
    def inst(self):
        return self._m
