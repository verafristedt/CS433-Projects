
import csv
import numpy as np
import pandas as pd # Only used for visualization
import matplotlib.pyplot as plt


# ***************************** Methods ******************************** #

def least_squares(y, tx):
    """
    Finds optimal weights and loss when computing least squares solution 
    when using normal equations.
    """
    
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Finds optimal weights and loss when computing least squares solution 
    when gradient descent.
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        w =  w - gamma * gradient
        
    
    loss = compute_mse(y,tx,w)  
    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Finds optimal weights and loss when computing least squares solution 
    when using stochastic gradient descent.
    """
    
    w = initial_w
    
    for n_iter,(y_, tx_) in enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters, shuffle=True)):
        
        stoch_gradient = compute_gradient(y_, tx_, w)
        loss = compute_mse(y_, tx_, w)
        w = w - gamma * stoch_gradient
        
        #print("Gradient Descent({bi}/{ti}): loss={l}, \nw={w}\n".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w=w))
        
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Finds optimal weights and loss when computing least squares solution with a 
    regularization term using normal equations.
    """
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters = 2000, gamma=0.000001):
    """
    Finds optimal weights and loss when using gradient descent on 
    logistic regression cost-function
    """
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute loss and gradient
        loss = compute_logistic_loss(y, tx, w)
        log_grad = compute_logistic_gradient(y, tx, w)
        # Calculate new w
        w = w - gamma * log_grad
        
    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters = 2000, gamma=0.000001):
    """
    Finds optimal weights and loss when using gradient descent on 
    logistic regression cost-function including a regularization term lambda.
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute loss and gradient while adding the regularization term
        loss = compute_logistic_loss(y, tx, w) + lambda_ * np.linalg.norm(w)
        log_grad = compute_logistic_gradient(y, tx, w) + lambda_ * w
        # Calculate new w
        w = w - gamma * log_grad

    return w, loss


# ***************************** Helpers ******************************** #

def compute_mse(y, tx, w):
    """
    Compute the Mean Squares Error
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """
    Computes the least squares gradient
    """
    e = y - tx.dot(w)
    
    return -(1/len(y))*((tx.T).dot(e))

    
def build_poly(x, degree):
    """
    Augmentes x to add extra features, with 1, x, x^2 ... x^degree
    """
    feature_matrix = np.ones((len(x), 1))

    # Degree is defined as highest power, so have to add one in range
    for degree in range(1, degree + 1):
        feature_matrix = np.c_[feature_matrix, np.power(x, degree)]
            
    return feature_matrix

    
def run_GD(y, tx, max_iters = 3000, gamma=0.008):
    """
    Helper function to easily run least squares gradient decent with our parameters
    """
    
    initial_w = np.array([0 for i in range(tx.shape[1])])
    return least_squares_GD(y, tx, initial_w, max_iters, gamma)


def run_SGD(y, tx, max_iters=3000, gamma = 0.0000001):
    """
    Helper function to easily run least squares stochastic gradient decent with our parameters
    """
    
    # Define the parameters of the algorithm.
    batch_size = 1

    # Initialization
    w_initial = np.zeros(tx.shape[1])

    # Start SGD.
    w, loss = least_squares_SGD(y, tx, w_initial, batch_size, max_iters, gamma)

    #print('Final loss: ', sgd_loss, '\nFinal weight vector:\n', sgd_w)
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    NB: COPIED FROM LAB-SESSION
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


def sigmoid(x):
    """Compute the sigmoid function"""
    return 1 / (1 + np.exp(-x))

def compute_logistic_loss(y, tx, w):
    """Compute the loss of the logistic model"""
    return np.sum(np.log(1 + np.exp(np.dot(tx, w)))) - np.dot(y.transpose(), np.dot(tx, w))


def compute_logistic_gradient(y, tx, w):
    """Calculate the gradient of the logistic function"""
    return (np.dot(tx.transpose(), sigmoid(np.dot(tx, w)) - y))

# ***************************** Cleaning Functions ******************************** #


def set_undefined_to_median(x):
    """ 
    Sets all values equal to -999 to the median of column.
    """
    x[x == -999] = np.nan
    x_median = np.nanmedian(x, axis = 0)
    indicies = np.where(np.isnan(x))
    x[indicies] = np.take(x_median, indicies[1])
    
    return x

def apply_log(x):
    """
    Takes the natrual logarithm of all columns that have nonnegative elements
    """
    neg_col= np.unique((np.where(x < 0 )[1]))
    neg_col = np.append(neg_col, 22)  # Add column containing jets
    
    pos_col = np.setdiff1d(np.arange(x.shape[1]), neg_col)
    x[:, pos_col] = np.log(1 + x[:,pos_col])
    
    return x
    
def apply_cosine_base(x):
    """
    Applies cosine base on all features that are angles
    """
    ang_col = [11,15,18, 20, 25, 28]
    x[:,ang_col] = np.cos(x[:,ang_col])
    
    return x


    
def remove_features(x, indicies):
    """
    Deletes all columns passed in indicies-parameter
    """
    
    return np.delete(x, indicies, axis = 1)


def clean_data(x):
    """
    Function to containarize the others
    """
    features_to_delete = [14, 15, 17, 18, 20]
    
    x = set_undefined_to_median(x)
    x = apply_log(x)
    x = apply_cosine_base(x)
    x = remove_features(x, features_to_delete)
    
    return x

# ***************************** Plots ******************************** #

def plot_hist(x):
    """
    Plots the distrubution of each future in x. 
    Uses matplotlib's histogram with 100 bins.
    """

    fig, ax = plt.subplots(10, 3, figsize = (15,30))
    fig.subplots_adjust(hspace = 0.5, wspace=0.2)

    ax = ax.ravel()
    for i in range(x.shape[1]):

        ax[i].hist(x[:,i], bins = 100)
        ax[i].set_title(i)

        
def plot_cross_validation_lambda(losses, filename = 'cross_validation_degree.png'):
    """
    Plots RMSE of each of tested lambdas.
    """
    
    fig = plt.figure(figsize=(8,6))

    plt.semilogx(losses.lambdas, losses.losses_tr, marker='o')
    plt.semilogx(losses.lambdas, losses.losses_te, marker='*')
    plt.title('Cross Validation - Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    plt.legend(['Training data', 'Testing data'])
    plt.grid()
    plt.savefig('./plots/' + filename)
    plt.show
    
    
def plot_cross_validation_degree(losses, filename = 'cross_validation_degree.png'):
    """
    Plots RMSE of each of tested degree.
    """
    fig = plt.figure(figsize=(8,6))
    
    plt.plot(losses.degree, losses.losses_tr, marker ='o')
    plt.plot(losses.degree, losses.losses_te, marker = '*')
    plt.title('Cross Validation - Degree')
    plt.xlabel('Degree')
    plt.ylabel('RMSE')
    plt.legend(['Training data', 'Testing data'])
    plt.xticks(np.arange(len(losses)))
    plt.grid()
    plt.ylim([0.74,1])
    plt.savefig('./plots/' + filename)
    plt.show()
    
    
def plot_cross_validation_gamma(losses, model='gd'): 
    """
    Plots RMSE of each of tested gamma.
    """
    fig = plt.figure(figsize=(8,6))
    
    plt.semilogx(losses.gammas, losses.losses_tr, marker='o')
    plt.semilogx(losses.gammas, losses.losses_te, marker='*')
    if model == 'gd':
        plt.title('Cross Validation - Gamma (GD)')
        plt.ylim(0.37, 0.5)
    else:
        plt.title('Cross Validation - Gamma (SGD)')
        plt.ylim(0.44, 0.7)
    plt.legend(['Training data', 'Testing data'])
    plt.xlabel('Gamma')
    plt.ylabel('MSE')
    plt.grid()
    plt.savefig('./plots/cross_validation_gamma_2.png')
    plt.show()


    
# ***************************** Cross Validation ******************************** #

    
    
def build_k_indices(y, k_fold, seed):
    """
    NB: COPIED FROM LABS
    Builds k-indicies used in cross-validation 
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
    Use kind to specify which algorithm to run.
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
        w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, inital_w, max_iters, gamma)
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
        Function for testing which regularization parameter to use in ridge regression.
    """
    seed = 100
    k_fold = 5    
    k_indices = build_k_indices(y, k_fold, seed)
    
    lambdas = np.logspace(-3,1,10)
    
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



# ***************************** Provided helper functions ******************************** #


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::25]
        input_data = input_data[::25]
        ids = ids[::25]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
            
            