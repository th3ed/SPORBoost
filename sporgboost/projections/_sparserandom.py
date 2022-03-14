from numba import njit
from sporgboost.utils import row_all
import numpy as np

@njit(cache=True)
def sparse_random(X, d, s):
    p = X.shape[1]

    thresh = 1 / (2 * s)
    out = np.zeros((p * d)).reshape((p, d))

    # Redraw if any mappings have all zero weights
    bad = row_all(out == 0)
    while bad.sum() > 0:
        draws =  np.random.uniform(0, 1, size=(p, d))
        for x, y in np.argwhere((draws < thresh) & bad.reshape((-1,1))):
            out[x, y] = -1

        for x, y in np.argwhere((draws > (1 - thresh)) & bad.reshape((-1,1))):
            out[x, y] = 1
        bad = row_all(out == 0)
    return out
