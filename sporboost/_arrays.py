from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def row_argmax(X):
    out = np.zeros(X.shape)
    for idx_y in range(0, X.shape[1]):
        out[np.argmax(X[:, idx_y]), idx_y] = 1
    return out

@njit(cache=True, fastmath=True)
def col_argmax(X):
    out = np.zeros(X.shape)
    for idx_x in range(0, X.shape[0]):
        out[idx_x, np.argmax(X[idx_x, :])] = 1
    return out

@njit(cache=True, fastmath=True)
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
    for idx_y in range(0, X.shape[1]):
        out[:, idx_y] = X[:, idx_y].cumsum()
    return out

@njit(cache=True, fastmath=True)
def row_mean(X, n=None):
    '''Numba optimized implementation of row-mean
    Args:
        X (Array): The array to mean(axis=0) over
    
    Returns:
        An array with the same shape as X with mean(axis=0) values
    '''
    if n is None:
        n = np.full(shape=(X.shape[0], 1), fill_value=1./X.shape[0])
    elif n.ndim == 1:
        # Reshape
        n = n.reshape((-1, 1))
    
    return (X * n).sum(axis=0) / n.sum()


@njit(cache=True, fastmath=True)
def row_norm(y):
    '''Normalizes a vector such that the row sum == 1
    
    Args:
        y (Array): 2-d vector to normalzie
    
    Returns:
        y vector normalized
    '''
    return(y / y.sum(axis=1).reshape((-1, 1)))

@njit(cache=True, fastmath=True)
def col_all(y):
    out = np.empty((y.shape[0]), dtype='bool')
    for idx_y in range(0, y.shape[0]):
        out[idx_y] = np.all(y[idx_y, :])
    return out

@njit(cache=True, fastmath=True)
def row_all(y):
    out = np.empty((y.shape[1]), dtype='bool')
    for idx_y in range(0, y.shape[1]):
        out[idx_y] = np.all(y[:, idx_y])
    return out

@njit(cache=True, fastmath=True)
def col_any(y):
    out = np.empty((y.shape[0]), dtype='bool')
    for idx_y in range(0, y.shape[0]):
        out[idx_y] = np.any(y[idx_y, :])
    return out

@njit(cache=True, fastmath=True)
def row_any(y):
    out = np.empty((y.shape[1]), dtype='bool')
    for idx_y in range(0, y.shape[1]):
        out[idx_y] = np.any(y[:, idx_y])
    return out

@njit(cache=True, fastmath=True)
def col_nunique(y):
    out = np.empty((y.shape[0]), dtype='int')
    for idx_y in range(0, y.shape[0]):
        out[idx_y] = np.unique(y[idx_y, :]).shape[0]
    return out

@njit(cache=True, fastmath=True)
def row_nunique(y):
    out = np.empty((y.shape[1]), dtype='int')
    for idx_y in range(0, y.shape[1]):
        out[idx_y] = np.unique(y[:, idx_y]).shape[0]
    return out

@njit(cache=True, fastmath=True)
def collapse_levels(X, y, n):
    y_ = {}
    n_ = {}

    # First pass: collapse any repeat values of X
    for idx_row in range(X.shape[0]):
        x_row = X[idx_row]
        y_row = y[idx_row, :]
        n_row = n[idx_row, :]
        if x_row not in y_:
            n_[x_row] = n_row.copy()
            y_[x_row] = y_row.copy() * n_row.copy()
        else:
            n_[x_row] += n_row
            y_[x_row] += y_row * n_row

    # Second pass: convert to arrays and apply sort
    X_ = np.array(list(y_.keys()))

    # For y and n we need to allocate a new array and fill it
    y2_ = np.empty(shape=(X_.shape[0], y.shape[1]))
    n2_ = np.empty(shape=(X_.shape[0], 1))
    for idx in range(y2_.shape[0]):
        x = X_[idx]
        y2_[idx, :] = y_[x]
        n2_[idx, :] = n_[x]

    idx = np.argsort(X_)
    return X_[idx], y2_[idx, :], n2_[idx]

@njit(cache=True)
def choice_replacement_weighted(X, y, D):
    idx = weighted_draws_replacement(np.arange(X.shape[0]), D, X.shape[0])
    return X[idx, :], y[idx, :]

@njit(cache=True)
def weighted_draws_replacement(a, p, n):
    out = np.empty(shape=(n), dtype='uint')

    for idx in range(n):
        # https://github.com/numba/numba/issues/2539
        out[idx] = a[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]
    return out