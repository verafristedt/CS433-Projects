# -*- coding: utf-8 -*-
"""function for plot."""
import matplotlib.pyplot as plt
import numpy as np



def plot_hist(x):

    fig, ax = plt.subplots(10, 3, figsize = (15,30))
    fig.subplots_adjust(hspace = 0.5, wspace=0.2)

    ax = ax.ravel()
    for i in range(x.shape[1]):

        ax[i].hist(x[:,i], bins = 100)
        ax[i].set_title(i)

        
def plot_losses(losses):
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(losses.degree, losses.losses_tr)
    plt.plot(loss.degree, losses.losses_te)
    plt.show()


        
def plot_cross_validation_lambda(losses):
    
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




   
    