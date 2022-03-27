import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def onehot_encode(y, levels = 0):
    '''One-hot encoding of a 1d vector of indices into a 2d array of 0-1 values
    
    Args:
        y (Array): 1-d array of index positions
        
    Returns:
        2-d array of indicator cols (1 indicating selected class)
    '''
    y_levels = levels if levels > 0 else np.max(y) + 1
    return(np.eye(y_levels)[y])