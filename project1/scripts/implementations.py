
import numpy as np
from scripts.proj1_helpers import *

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    np.random.seed(1) 

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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
        w =  w - gamma * gradient
        
    
    loss = compute_mse(y,tx,w)  
    return w, loss



def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    loss = 0
    w = initial_w
    
    for n_iter,(y_, tx_) in enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters, shuffle=True)):
        
        stoch_gradient = compute_gradient(y_, tx_, w)
        loss = compute_mse(y_, tx_, w)
        w = w - gamma * stoch_gradient
        
        #print("Gradient Descent({bi}/{ti}): loss={l}, \nw={w}\n".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w=w))
        
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w_ridge)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma=0.000001):
    """Logistic regression algorithm"""
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute loss and gradient
        loss = compute_logistic_loss(y, tx, w)
        log_grad = compute_logistic_gradient(y, tx, w)
        # Calculate new w
        w = w - gamma * log_grad
        
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma=0.000001):
    """Regularized logistic regression algorithm"""
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute loss and gradient while adding the regularization term
        loss = compute_logistic_loss(y, tx, w) + lambda_ * np.linalg.norm(w)
        log_grad = compute_logistic_gradient(y, tx, w) + lambda_ * w
        # Calculate new w
        w = w - gamma * log_grad

    return w, loss