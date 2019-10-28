
import numpy as np
import pandas as pd
from scripts.proj1_helpers import *
from scripts.implementations import *

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



def cross_validation(y, tx, k_indices, k, kind='gd', lambda_=0, degree=1, gamma_=0.001, max_iters=50):
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
    tx_tr = tx[indices_tr]
    tx_tr = build_poly(tx_tr, degree)
    
    # Test data
    y_te = y[indices_te]
    tx_te = tx[indices_te]
    tx_te = build_poly(tx_te, degree)
    
    
    if kind == 'gd':
        initial_w = np.zeros(tx_tr.shape[1])
        w, _ = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma_)
    elif kind == 'sgd':
        initial_w = np.zeros(tx_tr.shape[1])
        w, _ = least_squares_SGD(y_tr, tx_tr, initial_w, 1, max_iters, gamma_)
    elif kind == 'ls':
        w, _ = least_squares(y_tr, tx_tr)
    elif kind == 'ridge':
        w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    elif kind == 'log':
        w, _ = logistic_regression(y_tr, tx_tr, inital_w, max_iters, gamma)
    elif kind == 'reg_log':
        w, _ = re_logistic_regression(y_tr, tx_tr, lambda_, inital_w, max_iters, gamma)
    else:
        raise "Not valid value for 'kind'"
        
    loss_tr = compute_mse(y_tr, tx_tr, w)
    loss_te = compute_mse(y_te, tx_te, w)
    
    return loss_tr, loss_te

    
def cross_validation_least_squares(y, tx, max_degree):
    """
        Function for testing which degree we should use in our polynomial basis
        
    """
    seed = 10
    k_fold = 5
    k_indices = build_k_indices(y, k_fold, seed)
    
    degrees = np.arange(0,max_degree+1)
    
    losses_tr = []
    losses_te = []
    
    for degree in degrees:
        print('Currently at degree:', degree)
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='ls', degree = degree)
            
            temp_tr.append(np.sqrt(2*loss_tr))
            temp_te.append(np.sqrt(2*loss_te))
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'degree': degrees, 'losses_tr': losses_tr, 'losses_te': losses_te})
    

def cross_validation_ridge(y, tx, degree):
    """
        Function for testing which degree we should use in our polynomial basis
        
    """
    seed = 100
    k_fold = 5    
    k_indices = build_k_indices(y, k_fold, seed)
    
    lambdas = np.logspace(-3,3,7)
    
    losses_tr = []
    losses_te = []
    
    for lambda_ in lambdas:
        print(lambda_)
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='ridge', degree = degree, lambda_ = lambda_)
            
            temp_tr.append(np.sqrt(2*loss_tr))
            temp_te.append(np.sqrt(2*loss_te))
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'lambdas': lambdas, 'losses_tr': losses_tr, 'losses_te': losses_te})
    
    
def cross_validation_GD(y, tx, max_iters, degree):
    """
        Function for testing which gamma we should use in our gradient descent
    """
    seed = 10
    k_fold = 5
    k_indices = build_k_indices(y, k_fold, seed)
    
    gammas = np.logspace(-4,-2,20)
    
    losses_tr = []
    losses_te = []
    
    for gamma_ in gammas:
        print('Currently at gamma:', gamma_)
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='gd', degree=degree, gamma_=gamma_, max_iters=max_iters)
            
            temp_tr.append(loss_tr)
            temp_te.append(loss_te)
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'gammas': gammas, 'losses_tr': losses_tr, 'losses_te': losses_te})


def cross_validation_SGD(y, tx, max_iters, degree):
    """
        Function for testing which gamma we should use in our gradient descent
    """
    seed = 10
    k_fold = 5
    k_indices = build_k_indices(y, k_fold, seed)
    
    gammas = np.logspace(-5,-2,20)
    
    losses_tr = []
    losses_te = []
    
    for gamma_ in gammas:
        print('Currently at gamma:', gamma_)
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='sgd', degree=degree, gamma_=gamma_, max_iters=max_iters)
            
            temp_tr.append(loss_tr)
            temp_te.append(loss_te)
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'gammas': gammas, 'losses_tr': losses_tr, 'losses_te': losses_te})