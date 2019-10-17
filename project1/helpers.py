
import numpy as np


def standardize(x):
    """
    Returns the standardized version of x
    params:
        x - Raw Data
        
    returns:
        x - standardized
    """
    return (x - np.nanmean(x, axis = 0)) / (np.nanstd(x, axis = 0))
    
    
def build_tx(x):
    """
    Adds zero row for constants in front of x
    
    params: 
        x - standardized data
    returns:
        tx - augumented x matrix
    """
    
    return np.c_[np.zeros(len(x)), x]

