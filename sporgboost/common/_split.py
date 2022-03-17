from multiprocessing.sharedctypes import Value
from numba import njit, prange
import numpy as np
from .._arrays import row_cumsum, collapse_levels
from ._gini import gini_impurity

@njit(cache=True)
def best_split(X, y):
    col_split_gini = find_split(X, y)
    
    # Once all cols have been tested, determine which made the best split
    col_idx = np.argmin(col_split_gini[:,1])
    return (col_idx, col_split_gini[col_idx, 0])

@njit(parallel=False, cache=True)
def find_split(X, y):
    # Evaluate best split among each feature
    # 2d array where rows correspond to each col, 1st col is
    # split value and 2nd is gini
    col_split_gini = np.empty(shape = (X.shape[1], 2))
    for i in prange(0, X.shape[1]):
        col_split_gini[i, :] = _find_split_feat(X[:, i], y)
    return col_split_gini

@njit(cache=True)
def _find_split_feat(X, y):
    '''Determines where a split should be placed along a 1-d continuous feature col wrt y
    
    Args:
        X (Array): 1-d feature column to evalute for a split
        y (Array): 1-d response column
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    # Step 1: Get unique levels of X, counts and sums of y ordered by X
    # This solves problems later where we need the pair of X, y sorted
    # along X but also want to force the algorithm splits to consider
    # a split has to include all rows that match the level, not just
    # the one that mins gini
    X_, y_, n_ = collapse_levels(X, y)

    # Bail if X_ has only one level
    if X_.shape[0] == 1:
        return np.NaN, 999
    
    # Step 2: Compute the prediction for y if we made the split at the given row
    # Note we will remove the last level from these arrays as we can't split
    # inclusively on this and have obs in the right partition
    n_total = y.shape[0]
    y_total = y_.sum(axis=0)
    y_asc_cumsum = row_cumsum(y_)[:-1]
    y_desc_cumsum = y_total - y_asc_cumsum
    n_asc_cumsum = n_.cumsum()[:-1]
    n_desc_cumsum = n_total - n_asc_cumsum

    y_pred_left = (y_asc_cumsum / n_asc_cumsum.reshape((-1, 1)))
    y_pred_right = (y_desc_cumsum / n_desc_cumsum.reshape((-1, 1)))

    # Step 3: Compute gini impunity for left and right
    gini_left = gini_impurity(y_pred_left)
    gini_right = gini_impurity(y_pred_right)

    # Step 4: Compute weighted impunity
    weights_left = (n_asc_cumsum / n_total)
    weights_right = 1 - weights_left
    gini_split = gini_left * weights_left + gini_right * weights_right

    return _best_split_feat(X_[:-1], gini_split)

@njit(cache=True)
def _best_split_feat(X, gini):
    '''Proposes a split based on a sorted vector X and vector of weighted gini impunities
    
    Args:
        X (Array): A 1-d array of X values
        gini (Array): A 1-d array of weighted gini impunities assuming split occurs at X[idx]
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    idx_split = np.argmin(gini)
    # Taking mean of split values to align with sklearn
    x_split = X[idx_split:idx_split+2].mean()
    gini_split = gini[idx_split]

    return (x_split, gini_split)