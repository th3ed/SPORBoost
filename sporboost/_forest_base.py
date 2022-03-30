import numpy as np
from numba import njit
from ._arrays import col_argmax, col_all

# @njit(cache=True, fastmath=True)
def _predict_proba_forest(X, forest, n_classes, weights=None):
    # Init weights if they are None
    if weights is None:
        weights = np.full(shape=(len(forest)), fill_value = 1 / len(forest))

    # Scoring can be done in parallel in all cases
    out = np.zeros(shape=(X.shape[0], n_classes), dtype='float')
    for idx_tree in range(len(forest)):
        out += forest[idx_tree].predict(X) * weights[idx_tree]

    # Average prediction from all trees
    out /= weights.sum()

    return out

# @njit(cache=True, fastmath=True)
def _predict_forest(X, forest, n_classes, weights=None):
    probs = _predict_proba_forest(X, forest, n_classes, weights)
    votes = col_argmax(probs)
    return votes

@njit(cache=True, fastmath=True)
def _ada_misclassified(y_true, y_pred):
    return ~col_all(y_true == y_pred)

@njit(cache=True, fastmath=True)
def _ada_eta(misclassified, D):
    return np.sum(misclassified * D)

@njit(cache=True, fastmath=True)
def _ada_alpha(eta, n_classes):
    return 0.5 * np.log((1 - eta) / eta) + np.log(n_classes - 1.)

@njit(cache=True, fastmath=True)
def _ada_weight_update(y_true, y_pred, D, eta, miss, n_classes):
    alpha = _ada_alpha(eta, n_classes)

    # Check if we are upweighting or downweighting
    scalar = np.full(shape=(y_true.shape[0]), fill_value=alpha)
    scalar[~miss] *= -1.

    # Compute non-normalized weight updates
    D_new = D * np.exp(scalar)

    D_new /= D_new.sum()

    return D_new
