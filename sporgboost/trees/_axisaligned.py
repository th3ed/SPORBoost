import numpy as np
# from ..projections import identity
from ..common import best_split, gini_impurity
from ..common._predict import _predict
from ..utils import row_mean
from numba import njit
from numba.types import uint32, float64, deferred_type, optional
from numba.experimental import jitclass

@njit
def identity(X):
    return np.eye(X.shape[1])

# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////
node_type = deferred_type()

@jitclass([
    ('tree', optional(node_type)),
    ('n_classes', optional(uint32))
])
class AxisAlignedDecisionTree():
    def __init__(self):
        self.tree = None
        self.n_classes = None
        
    def fit(self, X, y):
        self.tree = _grow_tree_identity(X, y)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X)

# Node needs to be explicitly included in each tree type for numba
# to properly compile
@jitclass([
    ('value', optional(float64[:,:])),
    ('left', optional(node_type)),
    ('right', optional(node_type)),
    ('proj', optional(float64[:,:])),
    ('split', optional(float64)),
    ('n_classes', uint32)
])
class Node():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.value = None
        self.left = None
        self.right = None
        self.proj = None
        self.split = None

    def is_leaf(self):
        return self.value is not None
        

node_type.define(Node.class_type.instance_type)

@njit
def _grow_tree_identity(X, y):
    # Identity projection
    A = identity(X)
    X_ = X @ A

    col, split = best_split(X_, y)
    A_ = np.ascontiguousarray(A[:, col]).reshape((-1, 1))
    out = Node(y.shape[1])
    out.proj = A_
    out.split = split
    le = (X_[:, col] <= split)

    # Compute new split predictions
    pred_left = row_mean(y[le, :]).reshape((1, -1))
    pred_right = row_mean(y[~le, :]).reshape((1, -1))

    if gini_impurity(pred_left) == 0:
        # Return leaf value
        left = Node(out.n_classes)
        left.value = pred_left
        out.left = left
    else:
        # Grow another decision stump
        out.left = _grow_tree_identity(X[le, :], y[le, :])

    if gini_impurity(pred_right) == 0:
        # Return leaf value
        right = Node(out.n_classes)
        right.value = pred_right
        out.right = right
    else:
        out.right = _grow_tree_identity(X[~le, :], y[~le, :])
    
    return(out)