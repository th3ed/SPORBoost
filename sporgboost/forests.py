from ._forest_base import BaseRandomForest, BaseAdaBoost
from .trees import *
from numba.experimental import jitclass
from numba import uint32, optional

# @jitclass(
#     ('n_trees', uint32),
#     ('max_depth', optional(uint32)),
#     ('seed', uint32),
#     ('base_classifier', )
# )
class RandomForest(BaseRandomForest):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = AxisAlignedDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class AdaBoost(BaseAdaBoost):
    def __init__(self, n_trees = 500, max_depth = 1, seed = 1234):
        self.base_classifer = AxisAlignedDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class SPORF(BaseRandomForest):
    def __init__(self, d, s, n_trees = 500, max_depth = None, seed = 1234, **kwargs):
        self.base_classifer = SparseRandomDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, d=d, s=s)

class SPORGBoost(BaseAdaBoost):
    def __init__(self, d, s, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = SparseRandomDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, d=d, s=s)

class RotationalRandomForest(BaseRandomForest):
    def __init__(self, K, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = RotationalDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, K=K)

class RotBoost(BaseAdaBoost):
    def __init__(self, K, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = RotationalDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, K=K)
