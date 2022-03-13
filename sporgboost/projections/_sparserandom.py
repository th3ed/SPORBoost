from numba import njit
import numpy as np

@njit
def sparse_random(X, d, s):
    p = X.shape[1]
    # Step 1: Make random draws from uniform distribution
    draws =  np.random.uniform(0, 1, size=(p * d))

    # Step 2: Use random draws to set choices
    thresh = 1 / (2 * s)
    out = np.zeros((p * d))
    out[draws < thresh] = -1.
    out[draws > (1 - thresh)] = 1.
    return out.reshape((p, d))
