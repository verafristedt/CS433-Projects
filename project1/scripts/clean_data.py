#!/usr/bin/env python
# coding: utf-8

# # Cleaning of Data
# We use this file to clean the data. Still work in progress...  


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def remove_undefined_columns(x):
    """
    If more than 50% of the values in the column is -999, remove it.
    """
    bool_array = (np.count_nonzero(x==-999, axis=0) / x.shape[0]) > 0.5
    indicies = np.argwhere(bool_array == True).ravel()
    x = np.delete(x, indicies, axis=1)
    return x

def standardize(x):
    """
    Returns the standardized version of x
    params:
        x - Raw Data
        
    returns:
        x - standardized
    """
    return (x - np.nanmean(x, axis = 0)) / (np.nanstd(x, axis = 0))


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
    
    pos_col = np.setdiff1d(np.arange(30), neg_col)
    x[:, pos_col] = np.log(1 + x[:,pos_col])
    
    return x
    
def remove_outliers(x):
    """
    Set all outliers that are more than 2 std from the mean to the mean.
    """
    x_std = np.nanstd(x, axis = 0)
    x_mean = np.nanmean(x, axis = 0)
    indicies = np.where((x > (x_mean + 4*x_std)) | (x < x_mean + 4*x_std) )
    x[indicies] = np.take(x_mean, indicies[1])
    
    return x


def apply_cosine_base(x):
    """
    Applies cosine base on all features that are angles
    """
    ang_col = [11,15,18, 20, 25, 28]
    x[:,ang_col] = np.cos(x[:,ang_col])
    
    return x


def split_by_jets(y, x, jets_index=22):
    """
    Divide the set into three groups, based on PRI_jet_num
    """
    jet0 = x[x[:,jets_index] == 0]
    jet1 = x[x[:,jets_index] == 1]
    jet23 = x[(x[:,jets_index] == 2) | ((x[:,jets_index] == 3) )]
    
    # Delete the column containing jets
    jet0 = np.delete(jet0, jets_index, axis = 1)
    jet1 = np.delete(jet1, jets_index, axis = 1)
    jet23 = np.delete(jet23, jets_index, axis = 1)
    
    y0 = y[np.argwhere(x[:,jets_index] == 0)]
    y1 = y[np.argwhere(x[:,jets_index] == 1)]
    y23 = y[np.argwhere((x[:,jets_index] == 2) | (x[:,jets_index] == 3) )]

    return jet0, jet1, jet23, y0, y1, y23
    

    
def remove_features(x, indicies):
    return np.delete(x, indicies, axis = 1)

def clean_data(x):
    """
    Function to containarize the others
    """
    x = set_undefined_to_median(x)
    x = apply_log(x)
    x = apply_cosine_base(x)
    #x = standardize(x)
    x = remove_features(x,[14, 15, 17, 18, 20])
    
    return x

