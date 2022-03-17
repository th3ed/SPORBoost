# Disable numba jit so we can test coverage
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pytest
import sklearn.datasets
from sporgboost.preprocessing import onehot_encode, shuffle
from sporgboost.trees import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np

@pytest.fixture
def data_iris():
    return sklearn.datasets.load_iris(return_X_y = True)

def test_axisaligned(data_iris):
    np.random.seed(4321)

    X, y = data_iris

    y_ = onehot_encode(y)
    X_, y_ = shuffle(X, y_)
    X_train , y_train = X_[:-50, :], y_[:-50, :]
    X_test = X[-50:, :]

    model_aa = AxisAlignedDecisionTree()
    model_aa.fit(X_train, y_train)

    model_sklearn = DecisionTreeClassifier()
    model_sklearn.fit(X_train, y_train)

    assert np.all(model_aa.predict(X_train) == model_sklearn.predict(X_train))
    assert np.all(model_aa.predict(X_test) == model_sklearn.predict(X_test))

# Re-enable JIT
os.environ['NUMBA_DISABLE_JIT'] = ''