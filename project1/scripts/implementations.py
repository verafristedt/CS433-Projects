
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def least_squares(y, tx):
    """Calculates the least squares solution."""
    
    weights = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, weights)
    
    return loss, weights

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


