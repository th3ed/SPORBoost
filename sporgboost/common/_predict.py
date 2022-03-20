from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def _predict_tree(tree_pointer, tree_value, tree_split, tree_proj, X, n_classes):
    out = np.empty(shape=(X.shape[0], n_classes))

    # Traverse the tree, setting values as needed
    if tree_pointer not in tree_split:
        # Reached a leaf, return
        out[:,:] = 0.
        out[:, np.argmax(tree_value[tree_pointer].flatten())] = 1.
    else:
        # Decision Stump, keep parsing
        # Project X, then compare against split value
        X_ = np.dot(X, tree_proj[tree_pointer])
        le = X_.flatten() <= tree_split[tree_pointer]
        out[le, :] = _predict_tree(tree_pointer * 10, tree_value, tree_split, tree_proj, X[le], n_classes)
        out[~le, :] = _predict_tree(tree_pointer * 10 + 1, tree_value, tree_split, tree_proj, X[~le], n_classes)
    return out
