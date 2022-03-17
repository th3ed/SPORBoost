from ._forest_base import BaseRandomForest, BaseAdaBoost
from .trees import *

class RandomForest(BaseRandomForest):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = AxisAlignedDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class AdaBoost(BaseAdaBoost):
    def __init__(self, n_trees = 500, max_depth = 1, seed = 1234):
        self.base_classifer = AxisAlignedDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class SPORF(BaseRandomForest):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = SparseRandomDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class SPORGBoost(BaseAdaBoost):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = SparseRandomDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class RotationalRandomForest(BaseRandomForest):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = RotationalDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)

class RotBoost(BaseAdaBoost):
    def __init__(self, n_trees = 500, max_depth = None, seed = 1234):
        self.base_classifer = RotationalDecisionTree
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed)
