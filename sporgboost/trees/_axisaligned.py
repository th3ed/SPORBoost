from ..projections import identity
from ._predict import _predict
from ._growtree import _grow_tree

class AxisAlignedDecisionTree():
    def __init__(self):
        self.tree = None
        self.n_classes = None
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, identity)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X, self.n_classes)