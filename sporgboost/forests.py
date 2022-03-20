from numba.experimental import jitclass
from numba.types import uint32, int64, DictType
from .trees import *
from ._forest_base import _rf_fit, _predict_forest, _predict_proba_forest
import numpy as np

# @jitclass([
#     ('n_trees', int64),
#     ('max_depth', int64),
#     ('seed', uint32),
#     ('n_classes', int64),
#     ('forest', DictType(int64, AxisAlignedDecisionTree.class_type.instance_type))
# ])
class RandomForest():
    def __init__(self, n_trees = 500, max_depth = 10, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.n_classes = y.shape[1]
        self.forest = _rf_fit(X, y, self.n_trees, self.max_depth)

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

# class AdaBoost(BaseAdaBoost):
#     def __init__(self, n_trees = 500, max_depth = 1, seed = 1234):
#         self.base_classifer = AxisAlignedDecisionTree
#         super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

# class SPORF(BaseRandomForest):
#     def __init__(self, d, s, n_trees = 500, max_depth = None, seed = 1234, **kwargs):
#         self.base_classifer = SparseRandomDecisionTree
#         super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, d=d, s=s)

# class SPORGBoost(BaseAdaBoost):
#     def __init__(self, d, s, n_trees = 500, max_depth = None, seed = 1234):
#         self.base_classifer = SparseRandomDecisionTree
#         super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, d=d, s=s)

# class RotationalRandomForest(BaseRandomForest):
#     def __init__(self, K, n_trees = 500, max_depth = None, seed = 1234):
#         self.base_classifer = RotationalDecisionTree
#         super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, K=K)

# class RotBoost(BaseAdaBoost):
#     def __init__(self, K, n_trees = 500, max_depth = None, seed = 1234):
#         self.base_classifer = RotationalDecisionTree
#         super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, K=K)
