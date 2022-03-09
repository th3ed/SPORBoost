from sklearn.base import TransformerMixin
import numpy as np
from sklearn.decomposition import PCA
    
class Rotational(TransformerMixin):
    def __init__(self, p, K):
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # Input Validation
        if not (isinstance(p, int) and p > 0):
            raise ValueError("p must be a positive integer.")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer.")
            
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # ////////////////////////////////////////////
        # Assignments
        self.K = K
        self.p = p
        
        # Assign cols to each partition
        col_idx = np.random.permutation(np.arange(0, self.p))
        self.parts = np.array_split(col_idx, self.K)
        self.PCA = {i : PCA() for i in range(0, K)}
        self.col_idx = None
        
    def fit(self, X, y):
        for i in range(0, self.K):
            col_idx = self.parts[i]
            self.PCA[i].fit(X[:, col_idx])
            
        return self
        
    def transform(self, X):
        if self.col_idx is None:
            out = []
            for i in range(0, self.K):
                col_idx = self.parts[i]
                out.append(self.PCA[i].transform(X[:, col_idx]))
            return(np.concatenate(out, axis=1))
        
        col_idx = self.parts[self.col_idx]
        return self.PCA[self.col_pca_idx].transform(X[:, col_idx])[self.col_idx]
    
    def select(self, col_idx):
        pass