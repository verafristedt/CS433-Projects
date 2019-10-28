#!/usr/bin/env python
# coding: utf-8

# # Cleaning of Data
# We use this file to clean the data. Still work in progress...  


import numpy as np
import pandas as pd


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
    
    pos_col = np.setdiff1d(np.arange(x.shape[1]), neg_col)
    x[:, pos_col] = np.log(1 + x[:,pos_col])
    
    return x
    
def remove_outliers(x):
    """
    Set all outliers that are more than 2 std from the mean, to the mean.
    """
    x_std = np.nanstd(x, axis = 0)
    x_mean = np.nanmean(x, axis = 0)
    indicies = np.where((x > (x_mean + 4*x_std)) | (x < x_mean - 4*x_std) )
    x[indicies] = np.take(x_mean, indicies[1])
    
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

