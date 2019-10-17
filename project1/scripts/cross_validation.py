
import numpy as np
from costs import compute_mse

def build_k_indices(y, k_fold, seed):
    """
    
    params:
        y: Full y (dont divide into test and train)
        k_fold: Number of folds in cross-validation
        seed: For random generator
        
    return:
        k_indicies - Array with k_folds-subarrays containing indicies for the dataset
    
    """
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indicies, k, kind='gd' lambda_=0, degree=1):
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
    indices_tr = ind_tr.reshape(-1)
    
    # Training Data
    y_tr = y[indicies_tr]
    x_tr = x[indicies_tr]
    tx_tr = build_poly(x_tr,degree)    
    
    # Test data
    y_te = y[indicies_te]
    x_te = x[indicies_te]
    tx_te = build_poly(x_te, degree)
    
    
    if kind == 'gd':
        _, w = least_squares_GD(y, tx,initial_w, max_iters, gamma)
    else if kind == 'sgd':
        _, w = least_squares_SGD(y, tx,initial_w, max_iters, gamma)
    else if kind == 'ls':
        _, w = least_squares(y, tx)
    else if kind == 'ridge':
        _, w = ridge_regression(y, tx, lambda_)
    else if kind == 'log':
        _, w = logistic_regression(y, tx, inital_w, max_iters, gamma)
    else if kind == 'reg_log':
        _, w = re_logistic_regression(y, tx, lambda_, inital_w, max_iters, gamma)
    else:
        raise "Not valid value for 'kind'"
        
    loss_tr = compute_mse(y_tr, tx_tr, w)
    loss_te = compute_mse(y_te, tx_te, w)
    
    return loss_tr, loss_te


    
    
