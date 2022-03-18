from .trees import AxisAlignedDecisionTree
from .preprocessing import onehot_encode
from sklearn.base import BaseEstimator
from numba import prange
import numpy as np

class BaseForest(BaseEstimator):
    def __init__(self,
                 n_trees = 500,
                 seed = 1234,
                 max_depth = None,
                 **kwargs
                 ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self._forest = np.empty((self.n_trees), dtype='object')
        self.seed = seed

        # Initialize the classifiers
        for idx_tree in prange(self.n_trees):
            self._forest[idx_tree] = self.base_classifer(max_depth=self.max_depth, **kwargs)

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        np.random.seed(self.seed)

    def predict_proba(self, X):
        # Scoring can be done in parallel in all cases
        out = np.zeros(shape=(X.shape[0], self.n_classes), dtype='float')
        for idx_tree in prange(self.n_trees):
            out += self._forest[idx_tree].predict(X)

        # Average prediction from all trees
        out /= self.n_trees

        return out

    def predict(self, X):
        out = np.zeros(shape=(X.shape[0], self.n_classes))
        probs = self.predict_proba(X)
        return onehot_encode(np.argmax(probs, axis=1), levels=self.n_classes)

class BaseRandomForest(BaseForest):
    def fit(self, X, y):
        # Store metadata from training
        super().fit(X,y)

        # Random Forest trees can be fit in parallel
        for idx_tree in prange(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)
            self._forest[idx_tree].fit(X[idx_rows, :], y[idx_rows,:])

class BaseAdaBoost(BaseForest):
    def __init__(self,
                 n_trees = 500,
                 max_depth = 1,
                 seed = 1234,
                 **kwargs
                 ):
        super().__init__(n_trees = n_trees, max_depth = max_depth, seed = seed, **kwargs)

    def fit(self, X, y):
        # Store metadata from training
        super().fit(X,y)

        # Boosted trees must be fit sequentially
        # Give all samples equal weight initially
        D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

        final_n_trees = self.n_trees
        self.n_trees = 0
        for idx_tree in range(final_n_trees):
            # Draw a sample and fit a tree
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True, p=D)
            self._forest[idx_tree].fit(X[idx_rows, :], y[idx_rows,:])

            # Update weights based on forest errors
            self.n_trees += 1
            y_pred = self.predict(X)

            # Terminate early if all predictions match actuals
            if np.all(y_pred == y):
                break

            D = BaseAdaBoost._weight_update(y, y_pred, D)

        # Remove any unused trees
        self._forest = self._forest[:self.n_trees]
            

    @staticmethod
    def _misclassified(y_true, y_pred):
        return np.all(y_true == y_pred, axis=1)

    @staticmethod
    def _eta(misclassified, D):
        return np.sum(misclassified * D)

    @staticmethod
    def _alpha(eta):
        return 0.5 * np.log((1 - eta) / eta)

    @staticmethod
    def _weight_update(y_true, y_pred, D):
        miss = BaseAdaBoost._misclassified(y_true, y_pred)
        alpha = BaseAdaBoost._alpha(BaseAdaBoost._eta(miss, D))

        # Check if we are upweighting or downweighting
        scalar = np.full(shape=(y_true.shape[0]), fill_value=alpha)
        scalar[~miss] *= -1

        # Compute non-normalized weight updates
        D_new = D * np.exp(scalar)

        D_new /= D_new.sum()

        return D_new