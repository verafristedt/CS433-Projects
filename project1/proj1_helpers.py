# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


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
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

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

            
#**********************************************************************************************************************#

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

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


def compute_gradient(y, tx, w):
    """

    """
    e = y - tx.dot(w)
    
    return -(1/len(y))*((tx.T).dot(e))

    
    
def build_poly(x, degree):
    """
    Augmentes x to add extra features, with x, x^2 ... x^degree
    
    """
    feature_matrix = np.ones([len(x), 1])

    # Degree is defined as highest power, so have to add one in range
    for degree in range(1, degree + 1):
        feature_matrix = np.c_[feature_matrix, np.power(x, degree)]

    return feature_matrix