from numba import njit
import numpy as np

@njit
def gini_impunity(y):
    ''' Computes gini impunity for a given partition of data
    
    Args:
        y (Array): 2-d vector of normalized 
    '''
    return(1 - (y ** 2).sum(axis=1))

@njit
def weighted_gini(gini_left, gini_right):
    '''Compute weighted gini impunity given two arrays representing left, right partition impunity measures
    
    Args:
        gini_left (Array): 1-d array of gini impunities for the left partitioning at each index position
        gini_right (Array): 1-d array of gini impunities for the right partitioning at each index position
    
    Returns:
        A 1-d vector representing the weighted gini impunity when both partitions are considered
    '''
    n = gini_left.shape[0] + 1
    weight = np.arange(1, n) / n
    gini = gini_left * weight + gini_right * (1 - weight)
    return(gini)