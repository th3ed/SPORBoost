from .common import _predict_tree, _grow_tree
from .projections import identity, sparse_random, rotation
from sklearn.base import BaseEstimator

class BaseDecisionTree(BaseEstimator):
    def __init__(self, max_depth = None):
        self.tree = None
        self.n_classes = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        
    def predict(self, X):
        return _predict_tree(self.tree, X, self.n_classes)

class AxisAlignedDecisionTree(BaseDecisionTree):        
    def fit(self, X, y):
        super().fit(X, y)
        self.tree = _grow_tree(X, y, identity, max_depth = self.max_depth)
        
class SparseRandomDecisionTree(BaseDecisionTree):
    def __init__(self, d, s = 3, max_depth = None):
        self.d = d
        self.s = s
        super().__init__(max_depth=max_depth)
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, sparse_random, d=self.d, s=self.s, max_depth = self.max_depth)
        self.n_classes = y.shape[1]

class RotationalDecisionTree(BaseDecisionTree):
    def __init__(self, K, max_depth = None):
        self.K = K
        super().__init__(max_depth=max_depth)
        
    def fit(self, X, y):
        super().fit(X, y)
        self.tree = _grow_tree(X, y, rotation, K=self.K, max_depth = self.max_depth)