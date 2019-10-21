
import numpy as np
import pandas as pd
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
    tx_tr = tx[indices_tr]
    tx_tr = build_poly(tx_tr,degree)
    
    # Test data
    y_te = y[indices_te]
    tx_te = tx[indices_te]
    tx_te = build_poly(tx_te, degree)
    
    
    if kind == 'gd':
        w, _ = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
    elif kind == 'sgd':
        w, _ = least_squares_SGD(y_tr, tx_tr, initial_w, max_iters, gamma)
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


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w_ridge = np.linalg.solve(a, b)
    loss_ridge = (1/len(y))*np.transpose(y - tx@w_ridge).dot((y - tx@w_ridge))
    return loss_ridge, w_ridge


def plot_losses(losses):
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(losses.degree, losses.losses_tr)
    plt.plot(loss.degree, losses.losses_te)
    plt.show()


    
def cross_validation_least_squares(y, tx):
    """
        Function for testing which degree we should use in our polynomial basis
        
    """
    seed = 10
    k_fold = 5  
    k_indices = build_k_indices(y, k_fold, seed)
    
    degrees = np.arange(1,12,3)
    
    losses_tr = []
    losses_te = []
    
    for degree in degrees:
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='ls', degree = degree)
            
            temp_tr.append(np.sqrt(2*loss_tr))
            temp_te.append(np.sqrt(2*loss_te))
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'degree': degrees, 'losses_tr': losses_tr, 'losses_te': losses_te})
    

def cross_validation_ridge(y, tx):
    """
        Function for testing which degree we should use in our polynomial basis
        
    """
    seed = 10
    k_fold = 10    
    k_indices = build_k_indices(y, k_fold, seed)
    
    degrees = np.arange(4)
    
    losses_tr = []
    losses_te = []
    
    for degree in degrees:
        temp_tr = []
        temp_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, kind='ls', degree = degree)
            
            temp_tr.append(np.sqrt(2*loss_tr))
            temp_te.append(np.sqrt(2*loss_te))
        
        losses_tr.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
    
    return pd.DataFrame(data = {'degree': degrees, 'losses_tr': losses_tr, 'losses_te': losses_te})
    
    