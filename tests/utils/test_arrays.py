import pytest
import numpy as np
import pandas as pd
import sklearn.datasets
from sporgboost.utils import *
from sporgboost.preprocessing import onehot_encode

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

# def test_sort_pair(data_iris):
#     X, y = data_iris
#     X_ = X[:,0]
#     df = pd.DataFrame({'x' : X_, 'y' : y}).sort_values(by='x', ignore_index = True)
    
#     X_sorted, y_sorted = sort_pair(X_, y)
#     # df['y2'] = y_sorted
#     # print(df[['y', 'y2']][df.y != df.y2])
#     assert np.all(df['x'].values == X_sorted)
#     assert np.all(df['y'].values == y_sorted)

def test_collapse_levels(data_iris):
    X, y = data_iris
    X_ = X[:,0]
    y_ = onehot_encode(y)

    df = pd.get_dummies(pd.DataFrame({'x' : X_, 'y' : y}).astype({'y' : 'category'}))
    df_reduced = df.groupby('x').sum().reset_index()

    X_collapsed, y_collapsed, n_ = collapse_levels(X_, y_)
    assert np.all(df_reduced['x'] == X_collapsed)
    assert np.all(df_reduced.drop(['x'], axis=1) == y_collapsed)

