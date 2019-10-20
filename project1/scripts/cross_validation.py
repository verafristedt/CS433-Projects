
import numpy as np
from proj1_helpers import *
from implementations import *

def build_k_indices(y, k_fold, seed):
    """
    
    params:
        y: Full y (dont divide into test and train)
        k_fold: Number of folds in cross-validation
        seed: For random generator
        
    return:
        k_indices - Array with k_folds-subarrays containing indices for the dataset
    
    """
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, kind='gd', lambda_=0, degree=1):
    """
    General function that does cross validation for each of the methods.
    
    params:
    kind:
        gd - Gradient Decent
        sgd - Stochastic Gradient Decent
        ls - Least Squares
        ridge - Rigde Regression
        log - Logistic Regression
        reg_log - Regularized logistic regression
        
    returns:
        loss_tr, loss_te - Loss of training and testing for a given k
    """
    
    indices_te = k_indices[k]
    indices_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    indices_tr = indices_tr.reshape(-1)
    
    # Training Data
    y_tr = y[indices_tr]
    x_tr = tx[indices_tr]
    tx_tr = build_poly(x_tr,degree)    
    
    # Test data
    y_te = y[indices_te]
    x_te = tx[indices_te]
    tx_te = build_poly(x_te, degree)
    
    
    if kind == 'gd':
        _, w = least_squares_GD(y, tx,initial_w, max_iters, gamma)
    elif kind == 'sgd':
        _, w = least_squares_SGD(y, tx,initial_w, max_iters, gamma)
    elif kind == 'ls':
        _, w = least_squares(y, tx)
    elif kind == 'ridge':
        _, w = ridge_regression(y, tx, lambda_)
    elif kind == 'log':
        _, w = logistic_regression(y, tx, inital_w, max_iters, gamma)
    elif kind == 'reg_log':
        _, w = re_logistic_regression(y, tx, lambda_, inital_w, max_iters, gamma)
    else:
        raise "Not valid value for 'kind'"
        
    loss_tr = compute_mse(y_tr, tx_tr, w)
    loss_te = compute_mse(y_te, tx_te, w)
    
    return loss_tr, loss_te


    
def cross_validation_least_squares(y, tx):
    """
        Function for testing which degree we should use in our polynomial basis
        
    """
    seed = 100
    k_folds = 4    #Only use 10 if using full dataset (250000rows) else k_fold = 3
    k_indices = build_k_indices(y, k_folds, seed)
    
    degrees = np.arange(10)
    
    losses_tr = []
    losses_te = []
    
    for degree in degrees:
        temp_tr = []
        temp_te = []
        for k in range(k_folds):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='ls', degree = degree)
            
            temp_tr.append(loss_tr)
            temp_te.append(loss_te)
            
        losses_tr.append(np.average(temp_tr))
        losses_te.append(np.average(temp_te))
        
        
    print(losses_tr, losses_te)