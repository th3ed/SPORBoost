from ..common import gini_impurity, best_split
from ..utils import row_mean
from ..projections import identity
import numpy as np

class DecisionTree():
    def fit(self, X, y, proj = identity):
        self.tree = _grow_tree(X, y, proj)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X, self.n_classes)

def _grow_tree(X, y, proj):
    A = proj(X, y)
    X_ = X @ A

    col, split = best_split(X_, y)
    out = {'proj' : A[:, col], 'split' : split}
    le = (X_[:, col] <= split)

    # Compute new split predictions
    pred_left = row_mean(y[le, :]).reshape((1, -1))
    pred_right = row_mean(y[~le, :]).reshape((1, -1))

    if gini_impurity(pred_left) == 0:
        # Return leaf value
        out['left'] = pred_left
    else:
        # Grow another decision stump
        out['left'] = _grow_tree(X[le, :], y[le, :], proj)

    if gini_impurity(pred_right) <= .01:
        # Return leaf value
        out['right'] = pred_right
    else:
        pass
        out['right'] = _grow_tree(X[~le, :], y[~le, :], proj)
    
    return(out)

def _predict(tree, X, n_classes):
    # If we are at a leaf, return the value
    if isinstance(tree, np.ndarray):
        return tree

    # Decision Stump, keep parsing
    # Project X, then compare against split value
    X_ = X @ tree['proj']
    le = X_ <= tree['split']
    out = np.empty(shape=(X.shape[0], n_classes))
    out[le, :] = _predict(tree['left'], X[le], n_classes)
    out[~le, :] = _predict(tree['right'], X[~le], n_classes)
    
    return out