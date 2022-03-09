from sklearn.base import TransformerMixin

class Identity(TransformerMixin):
    def __init__(self):
        self.col_idx = None
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        if self.col_idx is None:
            return X
        return X[:, self.col_idx]
    
    def select(self, col_idx):
        self.col_idx = col_idx