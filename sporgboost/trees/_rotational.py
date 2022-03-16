from ..projections import rotation
from ._predict import _predict
from ._growtree import _grow_tree

class RotationalDecisionTree():
    def __init__(self, K):
        self.tree = None
        self.n_classes = None
        self.K = K
        
    def fit(self, X, y):
        self.tree = _grow_tree(X, y, rotation, K=self.K)
        self.n_classes = y.shape[1]

    def predict(self, X):
        return _predict(self.tree, X, self.n_classes)