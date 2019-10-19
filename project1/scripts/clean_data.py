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




def set_undefined_to_mean(x):
    """ 
    Sets all values equal to -999 to the mean of column.
    """
    x[x == -999] = np.nan
    x_mean = np.nanmean(x, axis = 0)
    indicies = np.where(np.isnan(x))
    x[indicies] = np.take(x_mean, indicies[1])
    
    return x



def remove_outliers(x):
    """
    Set all outliers that are more than 2 std from the mean to the mean.
    """
    x_std = np.nanstd(x, axis = 0)
    x_mean = np.nanmean(x, axis = 0)
    indicies = np.where((x > (x_mean + 2*x_std)) | (x < x_mean +2*x_std) )
    x[indicies] = np.take(x_mean, indicies[1])
    
    return x




def clean_data(x):
    """
    Function to containarize the two others.
    """
    x = remove_undefined_columns(x)
    x = set_undefined_to_mean(x)
    return x