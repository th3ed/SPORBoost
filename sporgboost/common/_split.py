from numba import njit, prange
import numpy as np
from ..utils import row_cumsum, row_norm, sort_pair
from ._gini import gini_impunity, weighted_gini

@njit
def best_split(X, y):
    col_split_gini = find_split(X, y)
    
    # Once all cols have been tested, determine which made the best split
    col_idx = np.argmin(col_split_gini[:,1])
    return (col_idx, col_split_gini[col_idx, 0])

@njit(parallel=True)
def find_split(X, y):
    # Evaluate best split among each feature
    # 2d array where rows correspond to each col, 1st col is
    # split value and 2nd is gini
    col_split_gini = np.empty(shape = (X.shape[1], 2))
    for i in prange(0, X.shape[1]):
        col_split_gini[i, :] = find_split_feat(X[:, i], y)
    return col_split_gini

@njit
def find_split_feat(X, y):
    '''Determines where a split should be placed along a 1-d continuous feature col wrt y
    
    Args:
        X (Array): 1-d feature column to evalute for a split
        y (Array): 1-d response column
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    # Step 1: Sort both vectors by X ascending
    X_sorted, y_sorted = sort_pair(X, y)
    
    # Step 2: Compute the prediction for y if we made the split at the given row
    y_pred_left = row_norm(row_cumsum(y_sorted[:-1, :]))
    
    # For right, flip the array to get reversed cumsums then flip it back to align to the original indicies
    y_pred_right = row_norm(row_cumsum(y_sorted[1:, :][::-1])[::-1])

    # Step 3: Calculate the gini impunity for each sub-partition
    gini_left = gini_impunity(y_pred_left)
    gini_right = gini_impunity(y_pred_right)
    
    # Step 4: Compute the weighted gini impunity for split of the parent node
    gini = weighted_gini(gini_left, gini_right)
    
    # Step 5: Return the value of X which had the largest decrease in gini impunity, along
    # with the actual impunity value to compare againt other features
    return _propose_split(X_sorted, gini)

@njit
def _propose_split(X, gini):
    '''Proposes a split based on a sorted vector X and vector of weighted gini impunities
    
    Args:
        X (Array): A 1-d array of X values
        gini (Array): A 1-d array of weighted gini impunities assuming split occurs at X[idx]
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    idx_split = np.argmin(gini)
    x_split = X[idx_split]
    gini_split = gini[idx_split]
    return (x_split, gini_split)