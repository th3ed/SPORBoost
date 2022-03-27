from numba.experimental import jitclass
from numba.types import uint32, int64, DictType
from .trees import *
from ._forest_base import _predict_forest, _predict_proba_forest, _ada_alpha, _ada_eta, _ada_misclassified, _ada_weight_update
import numpy as np
from ._arrays import choice_replacement_weighted

@jitclass([
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, AxisAlignedDecisionTree.class_type.instance_type)),
    ('n_classes', uint32)
])
class RandomForest():
    def __init__(self, n_trees = 500, max_depth = 10, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = AxisAlignedDecisionTree(self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

@jitclass([
    ('d', uint32),
    ('s', float64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, SparseRandomDecisionTree.class_type.instance_type)),
    ('n_classes', uint32)
])
class SPORF():
    def __init__(self, d, s, n_trees = 500, max_depth = 10, seed = 1234):
        self.d = d
        self.s = s
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = SparseRandomDecisionTree(self.d, self.s, self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

@jitclass([
    ('K', int64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, RotationalDecisionTree.class_type.instance_type)),
    ('n_classes', uint32)
])
class RotationalForest():
    def __init__(self, K, n_trees = 500, max_depth = 10, seed = 1234):
        self.K = K
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = RotationalDecisionTree(self.K, self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

@jitclass([
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, AxisAlignedDecisionTree.class_type.instance_type)),
    ('n_classes', uint32),
    ('alpha', float64[:])
])
class AdaBoost():
    def __init__(self, n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        self.n_classes = y.shape[1]

        # Initalize trees
        forest = {}
        alpha = {}

        # Boosted trees must be fit sequentially
        # Give all samples equal weight initially
        D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

        for idx_forest in range(self.n_trees):

            # Init and train a tree
            # Use weighted obs for training boosted trees
            forest[idx_forest] = AxisAlignedDecisionTree(self.max_depth)
            forest[idx_forest].fit(X, y, D)

            # Update weights based on forest errors
            y_pred = forest[idx_forest].predict(X)

            # Perform a weight update
            miss = _ada_misclassified(y, y_pred)
            eta = _ada_eta(miss, D)
            
            # Discard rules
            # https://github.com/scikit-learn/scikit-learn/blob/37ac6788
            # c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/ensemble/
            # _weight_boosting.py#L637
            if (eta <= 0.) or (eta >= 1. - (1.0 / y.shape[1])):
                # Tree is worse than random, break loop and return forest
                self.n_trees = idx_forest
                break
            
            # Tree is valid, we can update weights
            alpha[idx_forest] = _ada_alpha(eta, self.n_classes)
            D = _ada_weight_update(y, y_pred, D, eta, miss, self.n_classes)

        if self.n_trees > 0:
            self.forest = forest
            self.alpha = np.array(list(alpha.values()))

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes, weights=self.alpha)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes, weights=self.alpha)

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
