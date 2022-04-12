from sklearn.metrics import cohen_kappa_score
from .tree import *
from ._forest_base import _predict_forest, _predict_proba_forest, _ada_alpha, \
_ada_eta, _ada_misclassified, _ada_weight_update
import numpy as np
from sklearn.metrics import cohen_kappa_score

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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth}

    def set_params(self, max_depth = None):
        if max_depth is not None:
            self.max_depth = max_depth
        return self

class SPORF():
    def __init__(self, d_ratio=1., s_ratio=5., n_trees = 500, max_depth = 10, seed = 1234):
        self.d_ratio = d_ratio
        self.s_ratio = s_ratio
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
            forest[idx_forest] = SparseRandomDecisionTree(self.d_ratio, self.s_ratio, self.max_depth)
            forest[idx_forest].fit(X[idx_rows, :], y_[idx_rows,:])
        
        self.forest = forest

    def predict(self, X):
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'd_ratio' : self.d_ratio, 's_ratio' : self.s_ratio}

    def set_params(self, max_depth = None, s_ratio = None, d_ratio = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if s_ratio is not None:
            self.s_ratio = s_ratio
        if d_ratio is not None:
            self.d_ratio = d_ratio
        return self

class RotationalForest():
    def __init__(self, K=1, n_trees = 500, max_depth = 10, seed = 1234):
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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'K' : self.K}

    def set_params(self, max_depth = None, K = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if K is not None:
            self.K = K
        return self

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
            if (eta <= 0.) or (eta >= 1. - (1.0 / self.n_classes_)):
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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth}

    def set_params(self, max_depth = None):
        if max_depth is not None:
            self.max_depth = max_depth
        return self

class SPORBoost():
    def __init__(self, d_ratio=1., s_ratio=1., n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.s_ratio = s_ratio
        self.d_ratio = d_ratio

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
            forest[idx_forest] = SparseRandomDecisionTree(self.d_ratio, self.s_ratio, self.max_depth)
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
            if (eta <= 0.) or (eta >= 1. - (1.0 / self.n_classes_)):
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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'd_ratio' : self.d_ratio, 's_ratio' : self.s_ratio}

    def set_params(self, max_depth = None, s_ratio = None, d_ratio = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if s_ratio is not None:
            self.s_ratio = s_ratio
        if d_ratio is not None:
            self.d_ratio = d_ratio
        return self

class RotBoost():
    def __init__(self, K=1, n_trees = 500, max_depth = 1, seed = 1234):
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
            if (eta <= 0.) or (eta >= 1. - (1.0 / self.n_classes_)):
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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'K' : self.K}

    def set_params(self, max_depth = None, K = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if K is not None:
            self.K = K
        return self

class SPORBoost():
    def __init__(self, d_ratio=1., s_ratio=1., n_trees = 500, max_depth = 1, seed = 1234):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.s_ratio = s_ratio
        self.d_ratio = d_ratio

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
            # Fit a tree under each method, then select the one that has the lowest
            # misclassification rate
            tree_aa = AxisAlignedDecisionTree(self.max_depth)
            tree_aa.fit(X, y_, D)
            tree_sr = SparseRandomDecisionTree(self.d_ratio, self.s_ratio, self.max_depth)
            tree_sr.fit(X, y_, D)

            # Check which algo produced the lower misclassification rate
            y_pred_aa = tree_aa.predict(X)
            miss_aa = _ada_misclassified(y_, y_pred_aa)
            y_pred_sr = tree_sr.predict(X)
            miss_sr = _ada_misclassified(y_, y_pred_sr)

            # Select between the algos given the misclassification rate
            weights = np.array([miss_aa.sum(), miss_sr.sum()])
            weights = 1 - weights / weights.sum()
            draw = np.random.choice(['aa', 'sr'], size=1, p=weights.flatten())

            if draw == 'aa':
                forest[idx_forest] = tree_aa
                y_pred = y_pred_aa
                miss = miss_aa
            elif draw == 'sr':
                forest[idx_forest] = tree_sr
                y_pred = y_pred_sr
                miss = miss_sr

            # With the algo selected, set the tree, compute eta and update the weights
            eta = _ada_eta(miss, D)
            
            # Discard rules
            # https://github.com/scikit-learn/scikit-learn/blob/37ac6788
            # c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/ensemble/
            # _weight_boosting.py#L637
            if (eta <= 0.) or (eta >= 1. - (1.0 / self.n_classes_)):
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
        return np.argmax(_predict_forest(X, self.forest, self.n_classes_), axis=1)

    def predict_proba(self, X):
        return _predict_proba_forest(X, self.forest, self.n_classes_, weights=self.alpha)

    def score(self, X, y):
        pred = self.predict(X)
        return cohen_kappa_score(pred, y)

    def get_params(self, deep=True):
        return {'max_depth' : self.max_depth, 'd_ratio' : self.d_ratio, 's_ratio' : self.s_ratio}

    def set_params(self, max_depth = None, s_ratio = None, d_ratio = None):
        if max_depth is not None:
            self.max_depth = max_depth
        if s_ratio is not None:
            self.s_ratio = s_ratio
        if d_ratio is not None:
            self.d_ratio = d_ratio
        return self
