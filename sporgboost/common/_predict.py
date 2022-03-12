
from numba import njit
import numpy as np

@njit
def _predict(tree, X):
    # If we are at a leaf, return the value
    if tree.is_leaf():
        return tree.value

    # Decision Stump, keep parsing
    # Project X, then compare against split value
    X_ = np.dot(X, tree.proj)
    le = X_.flatten() <= tree.split
    out = np.empty(shape=(X.shape[0], tree.n_classes))
    out[le, :] = _predict(tree.left, X[le])
    out[~le, :] = _predict(tree.right, X[~le])
    
    return out