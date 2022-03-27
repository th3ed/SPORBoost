from ._tree_base import _predict_tree, _predict_proba_tree, _grow_tree
from .projections import identity, sparse_random, rotation
from .preprocessing import onehot_encode
from numba.experimental import jitclass
from numba.types import DictType, int64, float64, int64
import numpy as np

dt_spec = [
    ('tree_value', DictType(int64, float64[:,:])),
    ('tree_split', DictType(int64, float64)),
    ('tree_proj', DictType(int64, float64[:,:])),
    ('n_classes_', int64),
    ('classes_', int64[:]),
    ('max_depth', int64)
]

@jitclass(dt_spec)
class AxisAlignedDecisionTree():
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y, sample_weight=None):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y_, identity, self.max_depth, sample_weight)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def predict_proba(self, X):
        return _predict_proba_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth}

    def set_params(self, max_depth = None):
        if max_depth is not None:
            self.max_depth = max_depth
        return self

@jitclass(dt_spec + [
    ('d', int64),
    ('s', float64)
])
class SparseRandomDecisionTree():
    def __init__(self, d, s, max_depth = 10):
        self.d = d
        self.s = s
        self.max_depth = max_depth
        
    def fit(self, X, y, sample_weight=None):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y, sparse_random, self.max_depth, sample_weight, self.d, self.s)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def predict_proba(self, X):
        return _predict_proba_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'd' : self.d, 's' : self.s}

    def set_params(self, max_depth = None, s = None, d = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if s is not None:
            self.s = s
        if d is not None:
            self.d = d
        return self

@jitclass(dt_spec + [
    ('K', int64)
])
class RotationalDecisionTree():
    def __init__(self, K, max_depth = 10):
        self.K = K
        self.max_depth = max_depth
        
    def fit(self, X, y, sample_weight=None):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y, rotation, self.max_depth, sample_weight, self.K)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def predict_proba(self, X):
        return _predict_proba_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes_)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'K' : self.K}

    def set_params(self, max_depth = None, K = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if K is not None:
            self.K = K
        return self