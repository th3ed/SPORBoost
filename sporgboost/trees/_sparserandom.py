from this import d
from ..projections import sparse_random
from ._predict import _predict
from ._growtree import _grow_tree

# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////

# @jitclass([
#     ('tree', optional(node_type)),
#     ('n_classes', optional(uint32))
# ])
class SparseRandomDecisionTree():
    def __init__(self, d, s = 3):
        self.tree = None
        self.n_classes = None
        self.d = d
        self.s = s
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, sparse_random, d=self.d, s=self.s)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X, self.n_classes)