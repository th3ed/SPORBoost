from .common import _predict_tree, _grow_tree
from .projections import identity, sparse_random, rotation
from numba.experimental import jitclass
from numba.types import DictType, int64, float64, int64
import numpy as np

dt_spec = [
    ('tree_value', DictType(int64, float64[:,:])),
    ('tree_split', DictType(int64, float64)),
    ('tree_proj', DictType(int64, float64[:,:])),
    ('n_classes', int64),
    ('max_depth', int64)
]

@jitclass(dt_spec)
class AxisAlignedDecisionTree():
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y, sample_weight=None):
        self.n_classes = y.shape[1]
        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y, identity, self.max_depth, sample_weight)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes)

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
        self.n_classes = y.shape[1]
        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y, sparse_random, self.max_depth, sample_weight, self.d, self.s)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes)


@jitclass(dt_spec + [
    ('K', int64)
])
class RotationalDecisionTree():
    def __init__(self, K, max_depth = 10):
        self.K = K
        self.max_depth = max_depth
        
    def fit(self, X, y, sample_weight=None):
        self.n_classes = y.shape[1]
        if sample_weight is None:
            sample_weight = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])
        self.tree_value, self.tree_split, self.tree_proj = _grow_tree(X, y, rotation, self.max_depth, sample_weight, self.K)

    def predict(self, X):
        return _predict_tree(self.tree_value, self.tree_split, self.tree_proj, X, self.n_classes)