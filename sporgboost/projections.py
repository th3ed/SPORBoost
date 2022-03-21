from numba import njit
import numpy as np
import numpy as np
from numba import njit
from ._arrays import row_mean, col_all, row_argmax

@njit(cache=True, fastmath=True)
def identity(X):
    return np.eye(X.shape[1])

@njit(cache=True, fastmath=True)
def sparse_random(X, d, s):
    p = X.shape[1]

    thresh = 1 / (2 * s)
    out = np.zeros((p * d)).reshape((p, d))

    # Redraw if any mappings have all zero weights
    bad = col_all(out == 0)
    while bad.sum() > 0:
        draws =  np.random.uniform(0, 1, size=(p, d))
        for x, y in np.argwhere((draws < thresh) & bad.reshape((-1,1))):
            out[x, y] = -1

        for x, y in np.argwhere((draws > (1 - thresh)) & bad.reshape((-1,1))):
            out[x, y] = 1
        bad = col_all(out == 0)
    return out

@njit(cache=True, fastmath=True)
def pca(X):
    out = np.zeros(shape=(X.shape[1], X.shape[1]))

    # Step 1: Center data
    X_ = X - row_mean(X)

    # Get SVD decomposition for eigenvalues/vectors
    U, _, V = np.linalg.svd(X_, full_matrices=False)

    # SVD flip method
    max_abs_cols = row_argmax(np.abs(U)).astype(np.int64).flatten()
    signs = np.sign(np.diag(U[max_abs_cols, :]))
    U *= signs
    V *= signs.reshape((-1, 1))
    V = V.T

    # If N < M, by default numpy truncates the matrix of PCA weights
    # to N*M. We will fill in the remainer with zeros to not mess with
    # the shape rotation expects
    out[:, :V.shape[1]] = V
    return out

@njit(cache=True, fastmath=True)
def rotation(X, K):
    idx = np.arange(X.shape[1])
    np.random.shuffle(idx)
    parts = np.array_split(idx, K)

    out = np.zeros((X.shape[1], X.shape[1]))
    start = 0
    for idx_p in range(0, len(parts)):
        # Get index and relevant data
        p = parts[idx_p]
        end = start + len(p)
        X_ = X[:, p]

        # Project the data and save the weights
        out[p, start:end] = pca(X_)
        start = end

    return out