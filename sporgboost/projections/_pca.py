import numpy as np
from numba import njit
from ..utils import row_mean

@njit(cache=True)
def rotation(X, K):
    idx = np.arange(X.shape[1])
    np.random.shuffle(idx)
    parts = np.array_split(idx, K)

    out = np.zeros((X.shape[1], X.shape[1]))
    lens = np.array([len(x) for x in parts])
    ends = np.cumsum(lens)
    starts = ends - lens
    for idx_p in range(0, len(parts)):
        # Get index and relevant data
        p = parts[idx_p]
        start = starts[idx_p]
        end = start + len(p)
        X_ = X[:, p]

        # Project the data and save the weights
        out[p, start:end] = pca(X_)

    return out

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
