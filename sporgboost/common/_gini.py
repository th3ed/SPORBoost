from numba import njit
import numpy as np
from ..utils import row_cumsum

@njit
def gini_impunity(y):
    ''' Computes gini impunity for a given partition of data
    
    Args:
        y (Array): 2-d vector of normalized 
    '''
    n = np.arange(1, y.shape[0] + 1).reshape((-1,1))
    return 1 - ((row_cumsum(y) / n) ** 2).sum(axis=1)

@njit
def weighted_gini(gini_left, gini_right):
    '''Compute weighted gini impunity given two arrays representing left, right partition impunity measures
    
    Args:
        gini_left (Array): 1-d array of gini impunities for the left partitioning at each index position
        gini_right (Array): 1-d array of gini impunities for the right partitioning at each index position
    
    Returns:
        A 1-d vector representing the weighted gini impunity when both partitions are considered
    '''
    n = gini_left.shape[0] + 1
    weight = np.arange(1, n) / n
    gini = gini_left * weight + gini_right * (1 - weight)
    return(gini)


@njit(parallel=True)
def _collapse_levels(X, y):
    # Get unique levels of X
    X_unique = np.unique(X)
    y_agg = np.empty((X_unique.shape[0], y.shape[1]))
    n = np.empty((X_unique.shape[0]))
    for p in prange(0, X_unique.shape[0]):
        x_target = X_unique[p]
        y_agg[p, :] = row_mean(y[X == x_target])
        n[p] = y[X == x_target].shape[0]

    return X_unique, y_agg, row_cumsum(n)