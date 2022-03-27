# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pytest
import numpy as np
import pandas as pd
import sklearn.datasets
from sporboost._arrays import *
from sporboost.preprocessing import onehot_encode

@pytest.fixture
def data_iris():
    return sklearn.datasets.load_iris(return_X_y = True)

def test_row_cumsum(data_iris):
    X, _ = data_iris

    assert np.all(np.cumsum(X, axis=0) == row_cumsum(X))

def test_row_mean(data_iris):
    X, _ = data_iris

    assert np.all(X.mean(axis=0) == row_mean(X))

def test_row_norm(data_iris):
    _, y = data_iris
    y_ = onehot_encode(y)

    assert np.all((y_ / y_.sum(axis=1)[:, np.newaxis]) == row_norm(y_))

def test_collapse_levels(data_iris):
    X, y = data_iris
    X_ = X[:,0]
    y_ = onehot_encode(y)

    df = pd.get_dummies(pd.DataFrame({'x' : X_, 'y' : y}).astype({'y' : 'category'}))
    df_reduced = df.groupby('x').sum().reset_index()

    X_collapsed, y_collapsed, n_ = collapse_levels(X_, y_)
    assert np.all(df_reduced['x'] == X_collapsed)
    assert np.all(df_reduced.drop(['x'], axis=1) == y_collapsed)

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''