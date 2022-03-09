import numpy as np

def shuffle(X, y):
    '''Shuffle both X and y vectors in unison
    
    Args:
        X (Array): X array to shuffle
        y (Array): y array to shuffle
    
    Returns:
        X, y arrays, shuffled
    '''
    idx = np.random.permutation(y.shape[0])
    return(X[idx], y[idx])