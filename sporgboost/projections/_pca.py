import numpy as np
from numba import njit
from ..utils import row_mean

@njit(cache=True)
def pca(X):
    # Step 1: Center data
    X_ = X - row_mean(X)

    # Get SVD decomposition for eigenvalues/vectors
    U, _, V = np.linalg.svd(X_, full_matrices=False)

    # SVD flip method
    max_abs_cols = np.argmax(np.abs(U), axis=0).astype('int')
    signs = np.sign(np.diag(U[max_abs_cols, :]))
    U *= signs
    V *= signs.reshape((-1, 1))
    return V.T
