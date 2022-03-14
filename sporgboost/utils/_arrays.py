from numba import njit, prange
import numpy as np

@njit(parallel = False, cache=True)
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

@njit(cache=True)
def row_mean(X):
    '''Numba optimized implementation of row-mean
    Args:
        X (Array): The array to mean(axis=0) over
    
    Returns:
        An array with the same shape as X with mean(axis=0) values
    '''
    return X.sum(axis=0) / X.shape[0]

@njit(cache=True)
def row_norm(y):
    '''Normalizes a vector such that the row sum == 1
    
    Args:
        y (Array): 2-d vector to normalzie
    
    Returns:
        y vector normalized
    '''
    return(y / y.sum(axis=1).reshape((-1, 1)))

@njit(parallel = False, cache=True)
def col_all(y):
    out = np.empty((y.shape[0]), dtype='bool')
    for idx_y in prange(0, y.shape[0]):
        out[idx_y] = np.all(y[idx_y, :])
    return out

@njit(parallel = False, cache=True)
def row_all(y):
    out = np.empty((y.shape[1]), dtype='bool')
    for idx_y in prange(0, y.shape[1]):
        out[idx_y] = np.all(y[:, idx_y])
    return out

@njit(parallel = False, cache=True)
def col_any(y):
    out = np.empty((y.shape[0]), dtype='bool')
    for idx_y in prange(0, y.shape[0]):
        out[idx_y] = np.any(y[idx_y, :])
    return out

@njit(parallel = False, cache=True)
def row_any(y):
    out = np.empty((y.shape[1]), dtype='bool')
    for idx_y in prange(0, y.shape[1]):
        out[idx_y] = np.any(y[:, idx_y])
    return out

@njit(parallel = False, cache=True)
def col_nunique(y):
    out = np.empty((y.shape[0]), dtype='int')
    for idx_y in prange(0, y.shape[0]):
        out[idx_y] = np.unique(y[idx_y, :]).shape[0]
    return out

@njit(parallel = False, cache=True)
def row_nunique(y):
    out = np.empty((y.shape[1]), dtype='int')
    for idx_y in prange(0, y.shape[1]):
        out[idx_y] = np.unique(y[:, idx_y]).shape[0]
    return out

@njit(cache=True)
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