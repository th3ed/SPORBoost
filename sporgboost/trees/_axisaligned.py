from numba import njit
import numpy as np
from ..common import _predict_tree, _grow_tree

@njit(cache=True)
def identity(X):
    return np.eye(X.shape[1])

class AxisAlignedDecisionTree():
    def __init__(self):
        self.tree = None
        self.n_classes = None
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, identity)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict_tree(self.tree, X, self.n_classes)