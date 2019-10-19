
import numpy as np
from proj1_helpers import *


def least_squares(y, tx):
    """Calculates the least squares solution."""
    
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    
    return w, loss



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        w =  w - gamma*gradient
        
    
    loss = compute_mse(y,tx,w)  
    return w, loss


def run_gradient_decent(y, tx):
    
    max_iters = 500
    gamma = 0.000001
    
    initial_w = np.array([0.0005 for i in range(tx.shape[1])])
    return least_squares_GD(y, tx, initial_w, max_iters, gamma)