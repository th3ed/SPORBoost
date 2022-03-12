import numpy as np
from numba import njit

@njit
def identity(X, y):
    return np.eye(X.shape[1])