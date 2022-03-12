from sklearn.base import TransformerMixin
from numba.experimental import jitclass

class BaseProjection(TransformerMixin):
    def fit(self, X, y):
        raise NotImplementedError
    
    def transform(self, X):
        return X @ self.map

    def fit_map(self, X, y):
        self.fit(X, y)
        return self.transform(X), self.map