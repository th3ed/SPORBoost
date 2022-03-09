from numba import njit
import numpy as np

@njit
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
    for idx_col in range(0, X.shape[1]):
        out[:, idx_col] = X[:, idx_col].cumsum()
    return(out)


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