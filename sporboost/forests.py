from re import S
from numba.experimental import jitclass
from numba.types import uint32, int64, DictType
from .trees import *
from ._forest_base import _predict_forest, _predict_proba_forest, _ada_alpha, _ada_eta, _ada_misclassified, _ada_weight_update
import numpy as np

@jitclass([
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, AxisAlignedDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:])
])
class RandomForest():
    def __init__(self, n_trees = 500, max_depth = 10, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = AxisAlignedDecisionTree(self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y_[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth}

    def set_params(self, max_depth = None):
        if max_depth is not None:
            self.max_depth = max_depth
        return self

@jitclass([
    ('d', uint32),
    ('s', float64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, SparseRandomDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:])
])
class SPORF():
    def __init__(self, d, s, n_trees = 500, max_depth = 10, seed = 1234):
        self.d = d
        self.s = s
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = SparseRandomDecisionTree(self.d, self.s, self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y_[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

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

@jitclass([
    ('K', int64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, RotationalDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:])
])
class RotationalForest():
    def __init__(self, K, n_trees = 500, max_depth = 10, seed = 1234):
        self.K = K
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        # Initalize trees
        forest = {}

        for idx_forest in range(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

            # Init and train a tree
            forest[idx_forest] = RotationalDecisionTree(self.K, self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y_[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'K' : self.K}

    def set_params(self, max_depth = None, K = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if K is not None:
            self.K = K
        return self

@jitclass([
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, AxisAlignedDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:]),
    ('alpha', float64[:])
])
class AdaBoost():
    def __init__(self, n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

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
            forest[idx_forest].fit(X, y_, D)

            # Update weights based on forest errors
            y_pred = forest[idx_forest].predict(X)

            # Perform a weight update
            miss = _ada_misclassified(y_, y_pred)
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
            alpha[idx_forest] = _ada_alpha(eta, self.n_classes_)
            D = _ada_weight_update(y, y_pred, D, eta, miss, self.n_classes_)

        if self.n_trees > 0:
            self.forest = forest
            self.alpha = np.array(list(alpha.values()))

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth}

    def set_params(self, max_depth = None):
        if max_depth is not None:
            self.max_depth = max_depth
        return self

@jitclass([
    ('d', int64),
    ('s', float64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, SparseRandomDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:]),
    ('alpha', float64[:])
])
class SPORBoost():
    def __init__(self, d, s, n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.s = s
        self.d = d

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        # Initalize trees
        forest = {}
        alpha = {}

        # Boosted trees must be fit sequentially
        # Give all samples equal weight initially
        D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

        for idx_forest in range(self.n_trees):

            # Init and train a tree
            # Use weighted obs for training boosted trees
            forest[idx_forest] = SparseRandomDecisionTree(self.d, self.s, self.max_depth)
            forest[idx_forest].fit(X, y_, D)

            # Update weights based on forest errors
            y_pred = forest[idx_forest].predict(X)

            # Perform a weight update
            miss = _ada_misclassified(y_, y_pred)
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
            alpha[idx_forest] = _ada_alpha(eta, self.n_classes_)
            D = _ada_weight_update(y, y_pred, D, eta, miss, self.n_classes_)

        if self.n_trees > 0:
            self.forest = forest
            self.alpha = np.array(list(alpha.values()))

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

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

@jitclass([
    ('K', int64),
    ('n_trees', uint32),
    ('max_depth', int64),
    ('seed', uint32),
    ('forest', DictType(int64, RotationalDecisionTree.class_type.instance_type)),
    ('n_classes_', int64),
    ('classes_', int64[:]),
    ('alpha', float64[:])
])
class RotBoost():
    def __init__(self, K, n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.K = K

    def fit(self, X, y):
        if y.ndim == 2:
            self.n_classes_ = y.shape[1]
            y_ = y
        else:
            self.n_classes_ = np.max(y) + 1
            y_ = onehot_encode(y, levels = self.n_classes_)
        self.classes_ = np.arange(self.n_classes_)

        # Initalize trees
        forest = {}
        alpha = {}

        # Boosted trees must be fit sequentially
        # Give all samples equal weight initially
        D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

        for idx_forest in range(self.n_trees):

            # Init and train a tree
            # Use weighted obs for training boosted trees
            forest[idx_forest] = RotationalDecisionTree(self.K, self.max_depth)
            forest[idx_forest].fit(X, y_, D)

            # Update weights based on forest errors
            y_pred = forest[idx_forest].predict(X)

            # Perform a weight update
            miss = _ada_misclassified(y_, y_pred)
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
            alpha[idx_forest] = _ada_alpha(eta, self.n_classes_)
            D = _ada_weight_update(y, y_pred, D, eta, miss, self.n_classes_)

        if self.n_trees > 0:
            self.forest = forest
            self.alpha = np.array(list(alpha.values()))

    def predict(self, X):
        return _predict_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'K' : self.K}

    def set_params(self, max_depth = None, K = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if K is not None:
            self.K = K
        return self
