import numpy as np
from ..common import best_split, gini_impurity
from ..common._predict import _predict
from ..utils import row_mean
# from numba import njit
from numba.types import uint32, float64, deferred_type, optional
from numba.experimental import jitclass

# np.random.choice is not supported in numba
def sparse_random(X, d, s):
    p = X.shape[1]
    out =  np.random.choice(
        [-1., 0., 1.],
        size=(p, d),
        p=[1 / (2 * s),
        1 - (1 / s),
        1 / (2 * s)
        ]
    )

    # If any choices are all zeroes, redraw
    while np.any(out.sum(axis=0) == 0.):
        out = sparse_random(X, d, s)

    return out

# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////
node_type = deferred_type()

class SparseRandomDecisionTree():
    def __init__(self, d, s):
        self.tree = None
        self.n_classes = None
        self.d = d
        self.s = s
        
    def fit(self, X, y):
        self.tree = None
        self.n_classes = None
        self.tree = _grow_tree_sr(X, y, self.d, self.s)
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

def _grow_tree_sr(X, y, d, s):
    # Identity projection
    A = sparse_random(X, d, s)
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
        out.left = _grow_tree_sr(X[le, :], y[le, :], d, s)

    if gini_impurity(pred_right) == 0:
        # Return leaf value
        right = Node(out.n_classes)
        right.value = pred_right
        out.right = right
    else:
        out.right = _grow_tree_sr(X[~le, :], y[~le, :], d, s)
    
    return(out)
