from sklearn.base import TransformerMixin

class Identity(TransformerMixin):
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X