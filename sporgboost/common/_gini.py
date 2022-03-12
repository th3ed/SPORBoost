from numba import njit
import numpy as np
from ..utils import row_cumsum

@njit
def gini_impurity(y):
    ''' Computes gini impunity for a given partition of data
    
    Args:
        y (Array): 2-d vector of normalized 
    '''
    return 1 - (y ** 2).sum(axis=1)