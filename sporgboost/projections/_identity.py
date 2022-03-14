from numba import njit
import numpy as np

@njit
def identity(X):
    return np.eye(X.shape[1])