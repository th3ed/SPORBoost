from sklearn.base import TransformerMixin
from numba.experimental import jitclass
from numba import int16

@jitclass([
    ('col_idx', int16)
])
class Identity(TransformerMixin):
    def __init__(self):
        self.col_idx = -1
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        if self.col_idx == -1:
            return X
        # Force output to be 2d
        return X[:, self.col_idx : self.col_idx + 1]
    
    def select(self, col_idx):
        self.col_idx = col_idx