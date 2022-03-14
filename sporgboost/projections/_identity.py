from numba import njit
import numpy as np

@njit(cache=True)
def identity(X):
    return np.eye(X.shape[1])