# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pytest
import sklearn.datasets
from sporgboost.common import gini_impurity
from sporgboost.preprocessing import onehot_encode
import numpy as np

@pytest.fixture
def data_iris():
    return sklearn.datasets.load_iris(return_X_y = True)

def test_gini(data_iris):
    X, y = data_iris

    y_ = onehot_encode(y)

    # Partition off the first class of the data
    le = np.full(shape=(150), fill_value=False)
    le[:50] = True

    gini_left = gini_impurity(y_[le, :].mean(axis=0)[np.newaxis, :])
    gini_right = gini_impurity(y_[~le, :].mean(axis=0)[np.newaxis, :])
    
    assert gini_left == 0
    assert gini_right == 0.5

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''