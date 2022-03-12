from ..common import gini_impurity, best_split
from ..utils import row_mean
from ..projections import identity
import numpy as np
from sporgboost.projections import identity
from sporgboost.common import best_split, gini_impurity
from sporgboost.utils import row_mean
from numba import njit
from numba.types import boolean, uint32, float64, deferred_type, optional
from numba.experimental import jitclass

class DecisionTree():
    def fit(self, X, y, proj = identity):
        self.tree = _grow_tree(X, y, proj)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X)

node_type = deferred_type()

@jitclass([
    ('is_leaf', boolean),
    ('value', optional(float64[:,:])),
    ('left', optional(node_type)),
    ('right', optional(node_type)),
    ('proj', optional(float64[:,:])),
    ('split', optional(float64)),
    ('n_classes', uint32)
])
class Node():
    def __init__(self, value = None, left = None, right = None, proj = None, split = None, n_classes = None):       
        self.is_leaf = True if value is not None else False
        self.value = value
        self.left = left
        self.right = right
        self.proj = proj
        self.split = split
        self.n_classes = n_classes

node_type.define(Node.class_type.instance_type)

def _grow_tree(X, y, proj_func):
    A = proj_func(X, y)
    X_ = X @ A

    col, split = best_split(X_, y)
    A_ = np.ascontiguousarray(A[:, col]).reshape((-1, 1))
    out = Node(proj = A_, split = split, n_classes = y.shape[1])
    le = (X_[:, col] <= split)

    # Compute new split predictions
    pred_left = row_mean(y[le, :]).reshape((1, -1))
    pred_right = row_mean(y[~le, :]).reshape((1, -1))

    if gini_impurity(pred_left) == 0:
        # Return leaf value
        out.left = Node(value = pred_left, n_classes = out.n_classes)
    else:
        # Grow another decision stump
        out.left = _grow_tree(X[le, :], y[le, :], proj_func)

    if gini_impurity(pred_right) <= .01:
        # Return leaf value
        out.right = Node(value = pred_right, n_classes = out.n_classes)
    else:
        out.right = _grow_tree(X[~le, :], y[~le, :], proj_func)
    
    return(out)

@njit
def _predict(tree, X):
    # If we are at a leaf, return the value
    if tree.is_leaf:
        return tree.value

    # Decision Stump, keep parsing
    # Project X, then compare against split value
    X_ = np.dot(X, tree.proj)
    le = X_.flatten() <= tree.split
    out = np.empty(shape=(X.shape[0], tree.n_classes))
    out[le, :] = _predict(tree.left, X[le])
    out[~le, :] = _predict(tree.right, X[~le])
    
    return out
