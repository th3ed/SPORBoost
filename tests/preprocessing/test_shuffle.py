# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pytest
import sklearn.datasets
from sporgboost.preprocessing import shuffle
import numpy as np

@pytest.fixture
def data_iris():
    return sklearn.datasets.load_iris(return_X_y = True)

def test_shuffle(data_iris):
    X, y = data_iris

    # Shuffle the data, then re-sort to see if we have the same data as the start
    X_idx, y_ = shuffle(np.arange(0, X.shape[0]), y)
    X_, y_idx = shuffle(X, np.arange(0, X.shape[0]))

    assert np.all(y[X_idx] == y_)
    assert np.all(X[y_idx, :] == X_)

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''