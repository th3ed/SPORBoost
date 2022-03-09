from sklearn.base import TransformerMixin
import numpy as np
    
class SparseRandom(TransformerMixin):
    def __init__(self, p, d, s = 3):
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # Input Validation
        if not (isinstance(p, int) and d > 0):
            raise ValueError("p must be a positive integer.")
        if not (isinstance(d, int) and d > 0):
            raise ValueError("d must be a positive integer.")
        if s < 1:
            raise ValueError("s must be greater than 1")
            
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # Assignments
        self.p = p
        self.d = d
        self.s = s
        self.col_idx = None
        
        # sparse random projection mapping
        self.A = np.random.choice([-1, 0, 1],
                                  size=(self.p, self.d),
                                  p=[1 / (2 * self.s),
                                     1 - (1 / self.s),
                                     1 / (2 * self.s)
                                    ])
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        if self.col_idx is None:
            return(X @ self.A)
        return X @ self.A[:, self.col_idx]
    
    def select(self, col_idx):
        self.col_idx = col_idx