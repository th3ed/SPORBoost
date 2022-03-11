from sklearn.base import TransformerMixin
from numba.experimental import jitclass

@jitclass
class BaseProjection(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def transform(self, X):
        raise NotImplementedError
    
    def select(self, col_idx):
        raise NotImplementedError