import numpy as np
def onehot_encode(y):
    '''One-hot encoding of a 1d vector of indices into a 2d array of 0-1 values
    
    Args:
        y (Array): 1-d array of index positions
        
    Returns:
        2-d array of indicator cols (1 indicating selected class)
    '''
    return(np.eye(np.max(y) + 1)[y])