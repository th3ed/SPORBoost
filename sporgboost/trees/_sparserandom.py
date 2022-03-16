from ..common import _predict_tree, _grow_tree
from numba import njit
from sporgboost.utils import col_all
import numpy as np

@njit(cache=True)
def sparse_random(X, d, s):
    p = X.shape[1]

    thresh = 1 / (2 * s)
    out = np.zeros((p * d)).reshape((p, d))

    # Redraw if any mappings have all zero weights
    bad = col_all(out == 0)
    while bad.sum() > 0:
        draws =  np.random.uniform(0, 1, size=(p, d))
        for x, y in np.argwhere((draws < thresh) & bad.reshape((-1,1))):
            out[x, y] = -1

        for x, y in np.argwhere((draws > (1 - thresh)) & bad.reshape((-1,1))):
            out[x, y] = 1
        bad = col_all(out == 0)
    return out


class SparseRandomDecisionTree():
    def __init__(self, d, s = 3):
        self.tree = None
        self.n_classes = None
        self.d = d
        self.s = s
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, sparse_random, d=self.d, s=self.s)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict_tree(self.tree, X, self.n_classes)