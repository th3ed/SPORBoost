# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

from numba import prange
import numpy as np
from sporgboost.projections import sparse_random, identity, pca
import pytest
import sklearn.datasets
from sklearn.decomposition import PCA

@pytest.fixture
def data_iris():
    return sklearn.datasets.load_iris(return_X_y = True)

def test_identity():
    X = np.empty(shape=(1, 100))
    assert np.all(np.eye(X.shape[1]) == identity(X))

def test_sparse_random():
    np.random.seed(1234)
    tol = 1e-2
    X = np.empty(shape=(1,100))
    s = 3
    d = 50
    value_counts = np.zeros((3,))

    # Run n simulations and collect the running counts
    # of each outcome
    for i in prange(0,1000):
        A = sparse_random(X, d=d, s=s)
        value_counts[0] += (A == -1).sum()
        value_counts[1] += (A == 0).sum()
        value_counts[2] += (A == 1).sum()
        
    # Normalize outcomes
    value_dist = value_counts / value_counts.sum()

    # Estimate s again
    s_ = np.array(
        [1 / (2 * value_dist[0]),
        1 / (1 - value_dist[1]),
        1 / (2 * value_dist[2]),
    ])

    # Check values
    assert np.all(s_ > s - tol)
    assert np.all(s_ < s + tol)
    
def test_pca_tieout(data_iris):
    X, _ = data_iris
    V = pca(X)

    model_sklearn = PCA(svd_solver='full')
    model_sklearn.fit(X)

    X_pca = (X - X.mean(axis=0)) @ V

    # Weights can be flipped based on the SVD engine, flip
    # signs and check
    X_pca_sklearn =  model_sklearn.transform(X)
    return np.all(np.abs(X_pca - X_pca_sklearn) < 1e-8)

def test_pca_centered(data_iris):
    # This test confirms that PCA series transformed
    # on original or de-meaned data only differ by a constant
    # which can be baked into the node split
    X, _ = data_iris

    V = pca(X)
    centered_pca = (X - X.mean(axis=0)) @ V
    uncentered_pca = X @ V

    assert np.all(np.var(centered_pca - uncentered_pca, axis=0) < 1e-16)

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''