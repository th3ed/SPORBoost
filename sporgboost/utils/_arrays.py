from numba import njit, prange
import numpy as np

@njit(parallel = True)
def row_cumsum(X):
    '''Numba optimized implementation of row-cumsum
    This function is needed to allow broader functions such as find_split compile with numba. Numba
    supports cumsum() but only with no axis arguments, this function essentially loops over the
    columns to achieve a cumsum(axis=0)
    
    Args:
        X (Array): The array to cumsum(axis=0) over
    
    Returns:
        An array with the same shape as X with cumsum(axis=0) values
    '''
    out = np.empty(X.shape)
    for idx_col in prange(0, X.shape[1]):
        out[:, idx_col] = X[:, idx_col].cumsum()
    return out

@njit
def row_mean(X):
    '''Numba optimized implementation of row-mean
    Args:
        X (Array): The array to mean(axis=0) over
    
    Returns:
        An array with the same shape as X with mean(axis=0) values
    '''
    return X.sum(axis=0) / X.shape[0]

@njit
def row_norm(y):
    '''Normalizes a vector such that the row sum == 1
    
    Args:
        y (Array): 2-d vector to normalzie
    
    Returns:
        y vector normalized
    '''
    return(y / y.sum(axis=1).reshape((-1, 1)))

@njit
def sort_pair(X, y):
    '''Sort two 1-d vectors X and y according to X
    
    Args:
        X (Array): 1-d array of X values to sort X and y on
        y (Array): 1-d array of y values to be sorted based on X
        
    Returns:
        X, y pair of sorted values
    '''
    idx_sorted = np.argsort(X)
    X_sorted, y_sorted = X[idx_sorted], y[idx_sorted]
    return(X_sorted, y_sorted)

@njit
def collapse_levels(X, y):
    # Get unique levels of X, and init outputs
    X_ = np.unique(X)
    y_ = np.empty(shape=(X_.shape[0], y.shape[1]), dtype='float')
    n_ = np.empty(shape=(X_.shape[0]), dtype='int')

    for idx_x in prange(0, X_.shape[0]):
        match = (X == X_[idx_x])
        y_[idx_x, :] = y[match, :].sum(axis=0)
        n_[idx_x] = match.sum()
    return X_, y_, n_