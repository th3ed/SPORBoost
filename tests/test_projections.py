# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

from numba import prange
import numpy as np
from sporgboost.projections import sparse_random

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
    
test_sparse_random()

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''