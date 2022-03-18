from numba import njit
import numpy as np

# @njit(cache=True)
def _predict_tree(tree, X, n_classes):
    out = np.empty(shape=(X.shape[0], n_classes))

    # If we are at a leaf, return the value
    if tree.is_leaf():
        # Changed scoring value to use voting
        out[:,:] = 0.
        out[:, np.argmax(tree.value)] = 1.
    else:
        # Decision Stump, keep parsing
        # Project X, then compare against split value
        X_ = np.dot(X, tree.proj)
        le = X_.flatten() <= tree.split
        out[le, :] = _predict_tree(tree.left, X[le], n_classes)
        out[~le, :] = _predict_tree(tree.right, X[~le], n_classes)
    return out