from sporgboost.trees import AxisAlignedDecisionTree
from .preprocessing import onehot_encode
from sklearn.base import BaseEstimator
import numpy as np
from numba import njit, prange

# @njit(cache=True, fastmath=True)
def _predict_proba_forest(X, forest, n_classes):
    # Scoring can be done in parallel in all cases
    out = np.zeros(shape=(X.shape[0], n_classes), dtype='float')
    for idx_tree in range(len(forest)):
        out += forest[idx_tree].predict(X)

    # Average prediction from all trees
    out /= len(forest)

    return out

# @njit(cache=True, fastmath=True)
def _predict_forest(X, forest, n_classes):
    out = np.zeros(shape=(X.shape[0], n_classes))
    probs = _predict_proba_forest(X, forest, n_classes)
    return onehot_encode(np.argmax(probs, axis=1).astype('int'), levels=n_classes)

# Can't cache parallel functions
# @njit(cache=False, fastmath=True)
def _rf_fit(X, y, n_trees, max_depth):
    # Initalize trees
    forest = {}

    for idx_forest in range(n_trees):
        # Draw a bootstrapped sample
        idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)

        # Init and train a tree
        forest[idx_forest] = AxisAlignedDecisionTree(max_depth)
        forest[idx_forest].fit(X[idx_rows, :], y[idx_rows,:])
    
    return forest

@njit(cache=True, fastmath=True)
def _ada_fit(X, y, base_classifier, n_trees, seed, *args):
    np.random.seed(seed)

    # Initalize trees
    forest = {}
    n_classes = y.shape[1]

    # Boosted trees must be fit sequentially
    # Give all samples equal weight initially
    D = np.full(shape=(X.shape[0]), fill_value=1/X.shape[0])

    for idx_forest in range(n_trees):
        invalid_tree = True
        attempt = 0
        max_attempts = 5
        while invalid_tree and (attempt < max_attempts):
            attempt += 1

            # Draw a sample
            idx_rows = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True, p=D)

            # Init and train a tree
            forest[idx_forest] = base_classifier(*args)
            forest[idx_forest].fit(X[idx_rows, :], y[idx_rows,:])

            # Update weights based on forest errors
            y_pred = _predict_forest(X, forest, n_classes)

            # Perform a weight update
            miss = _ada_misclassified(y, y_pred)
            eta = _ada_eta(miss, D)

            # Discard tree if eta=0 or eta>0.5
            if eta == 0. or eta >= 0.5:
                continue
            
            # Tree is valid, we can update weights and break the loop
            invalid_tree + False

            D = _ada_weight_update(y, y_pred, D, eta, miss)

        if invalid_tree:
            print("Terminated after {max_attempts} candidate trees were rejected")
            continue

    return forest

@njit(cache=True, fastmath=True)
def _ada_misclassified(y_true, y_pred):
    return np.all(y_true == y_pred, axis=1)

@njit(cache=True, fastmath=True)
def _ada_eta(misclassified, D):
    return np.sum(misclassified * D)

@njit(cache=True, fastmath=True)
def _ada_alpha(eta):
    return 0.5 * np.log((1 - eta) / eta)

@njit(cache=True, fastmath=True)
def _ada_weight_update(y_true, y_pred, D, eta, miss):
    alpha = _ada_alpha(eta)

    # Check if we are upweighting or downweighting
    scalar = np.full(shape=(y_true.shape[0]), fill_value=alpha)
    scalar[~miss] *= -1

    # Compute non-normalized weight updates
    D_new = D * np.exp(scalar)

    D_new /= D_new.sum()

    return D_new
