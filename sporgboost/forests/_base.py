from ..trees import AxisAlignedDecisionTree
from sklearn.base import BaseEstimator
from numba import prange
import numpy as np

class BaseForest(BaseEstimator):
    def __init__(self,
                 n_trees = 10,
                 seed = 1234
                 ):
        self.base_classifer = AxisAlignedDecisionTree
        self.n_trees = n_trees
        self._forest = np.empty((self.n_trees))
        self.seed = seed

        # Initialize the classifiers
        for idx_tree in prange(self.n_trees):
            self._forest[idx_tree] = self.base_classifer()

    def fit(self, X, y):
        self.n_classes = y.shape[1]

    def predict_proba(self, X):
        # Scoring can be done in parallel in all cases
        out = np.zeros(shape=(X.shape[0], self.n_classes))
        for idx_tree in prange(self.n_trees):
            out += self._forest.predict(X)

        # Average prediction from all trees
        out /= self.n_trees

        return out

    def predict(self, X):
        out = np.zeros(shape=(X.shape[0], self.n_classes))
        probs = self.predict(X)
        out[np.argmax(probs, axis=1)] = 1
        return out

class BaseRandomForest(BaseForest):
    def fit(self, X, y):
        # Store metadata from training
        super().fit(X,y)

        # Random Forest trees can be fit in parallel
        np.random.seed(self.seed)

        for idx_tree in prange(self.n_trees):
            # Draw a bootstrapped sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)
            self._forest[idx_tree].fit(X[idx_rows, :], y[idx_rows,:])

class BaseAdaBoost(BaseForest):
    def fit(self, X, y):
        # Boosted trees must be fit sequentially
        np.random.seed(self.seed)

        # Give all samples equal weight initially
        D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

        for idx_tree in range(self.n_trees):
            # Draw a sample and fit a tree
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True, p=D)
            self._forest[idx_tree].fit(X[idx_rows, :], y[idx_rows,:])

            # Update weights based on forest errors
            y_pred = self.predict(X)
            D = BaseAdaBoost._weight_update(y, y_pred, D)
            

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