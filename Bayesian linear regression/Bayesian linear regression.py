# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:01:58 2019

@author: sonitabh
"""
#%% Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm
import math

#%% Initialization

if __name__ == "__main__":
    
    A = 1  # Set equal to 1 for Identity, 2 for Monomial, 3 for Fourier and 4 for Legendre: A = {1,2,3,4}
    M = 8  # M = {1,2,3,4,5,6,7,8}: Set any value upto 8 depending on number of features needed
    N = 500  # N is the number of training points
    
    alpha = 5
    beta = 0.1

    
    X = 2*lhs(1,N)
    e = np.random.normal(0,math.sqrt(0.5),N)
    e = np.reshape(e,(N,1))
    y = np.exp(X)*np.sin(2*math.pi*X) + e
    
    
    if A == 1:
        # Identity Basis
        phi = X
    elif A == 2:
        # Monomial Basis (M = 8)
        phi = np.hstack((np.ones((N,1)) ,X, X**2, X**3, X**4, X**5, X**6, X**7, X**8))
        phi = phi[:,0:(M+1)]
    elif A == 3:
        # Fourier Basis (M = 8)
        phi = np.hstack((np.zeros((N,1)), np.ones((N,1)), np.sin((np.pi)*X), np.cos((np.pi)*X), np.sin(2*(np.pi)*X), np.cos(2*(np.pi)*X), np.sin(3*(np.pi)*X), np.cos(3*(np.pi)*X), np.sin(4*(np.pi)*X), np.cos(4*(np.pi)*X), np.sin(5*(np.pi)*X), np.cos(5*(np.pi)*X), np.sin(6*(np.pi)*X), np.cos(6*(np.pi)*X),np.sin(7*(np.pi)*X), np.cos(7*(np.pi)*X),np.sin(8*(np.pi)*X), np.cos(8*(np.pi)*X)))
        phi = phi[:,0:2*(M+1)]
    elif A == 4:    
        # Legendre Basis
        phi = np.hstack((np.ones((N,1)), X, 0.5*(3*(X**2) - 1), 0.5*(5*(X**3) - 3*X), 0.125*(35*(X**4) - 30*(X**2) + 3), 0.125*(63*(X**5) - 70*(X**3) + 15*X), 0.0625*(231*(X**6) - 315*(X**4) + 105*(X**2) - 5), 0.0625*(429*(X**7) - 693*(X**5) + 315*(X**3) - 35*X), (1/128)*(6435*(X**8) - 12012*(X**6) + 6930*(X**4) - 1260*(X**2) + 35)))
        phi = phi[:,0:(M+1)]
    else:
        phi = 0
        print("Input a valid A correponding to the right basis")
        
        
#%% MLE weight calculation
        
    jitter = 1e-8
    xTx_inv = np.linalg.inv(np.matmul(phi.T,phi) + jitter) 
    xTy_mle = np.matmul(phi.T,y)
    w_mle = np.matmul(xTx_inv, xTy_mle)
    
    
#%% MAP weight calculation
    
    Lambda = np.matmul(phi.T,phi) + (beta/alpha)*np.eye(phi.shape[1])
    Lambda_inv = np.linalg.inv(Lambda)
    xTy_map = np.matmul(phi.T, y)
    mu = np.matmul(Lambda_inv, xTy_map)
    w_map = mu
    
    
#%% Testing
    
    X_star = np.linspace(0,2,N)[:,None]
    
    if A == 1:
        # Identity Basis
        phi_star = X_star
    elif A == 2:
        # Monomial Basis (M = 8)
        phi_star = np.hstack((np.ones((N,1)) ,X_star, X_star**2, X_star**3, X_star**4, X_star**5, X_star**6, X_star**7, X_star**8))
        phi_star = phi_star[:,0:(M+1)]
    elif A == 3:
        # Fourier Basis (M = 8)
        phi_star = np.hstack((np.zeros((N,1)), np.ones((N,1)), np.sin((np.pi)*X_star), np.cos((np.pi)*X_star), np.sin(2*(np.pi)*X_star), np.cos(2*(np.pi)*X_star), np.sin(3*(np.pi)*X_star), np.cos(3*(np.pi)*X_star), np.sin(4*(np.pi)*X_star), np.cos(4*(np.pi)*X_star), np.sin(5*(np.pi)*X_star), np.cos(5*(np.pi)*X_star), np.sin(6*(np.pi)*X_star), np.cos(6*(np.pi)*X_star), np.sin(7*(np.pi)*X_star), np.cos(7*(np.pi)*X_star), np.sin(8*(np.pi)*X_star), np.cos(8*(np.pi)*X_star)))
        phi_star = phi_star[:,0:2*(M+1)]
    elif A == 4:    
        # Legendre Basis
        phi_star = np.hstack((np.ones((N,1)), X_star, 0.5*(3*(X_star**2) - 1), 0.5*(5*(X_star**3) - 3*X_star), 0.125*(35*(X_star**4) - 30*(X_star**2) + 3), 0.125*(63*(X_star**5) - 70*(X_star**3) + 15*X_star), 0.0625*(231*(X_star**6) - 315*(X_star**4) + 105*(X_star**2) - 5), 0.0625*(429*(X_star**7) - 693*(X_star**5) + 315*(X_star**3) - 35*X_star), (1/128)*(6435*(X_star**8) - 12012*(X_star**6) + 6930*(X_star**4) - 1260*(X_star**2) + 35)))
        phi_star = phi_star[:,0:M+1]
    else:
        phi_star = 0
        print("Input a valid A correponding to the right basis")
        
    Y_star_mle = np.matmul(phi_star,w_mle)
    Y_star_map = np.matmul(phi_star,w_map)
    
   
#%% samples from the predictive posterior
    
    num_samples = 500;
    mean_star = np.matmul(phi_star, w_map)
    var_star = 1.0/alpha + np.matmul(phi_star, np.matmul(Lambda_inv, phi_star.T))
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

#%% Plot the Results
    
    plt.figure(1, figsize=(24,18))
    plt.subplot(1,2,1)
    plt.plot(X_star, Y_star_mle, linewidth=3.0, label = 'MLE')
    plt.plot(X_star, Y_star_map, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i,:], 'k', linewidth=0.05)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    
    
    
    
    if A == 1:
        # Identity Basis
        p = X_star
        l = ('x')
    elif A == 2:
        # Monomial Basis (M = 8)
        p = np.hstack((np.ones((N,1)) ,X_star, X_star**2, X_star**3, X_star**4, X_star**5, X_star**6, X_star**7, X_star**8))
        p = p[:,0:5]
        l = ('1','x','x^2','x^3','x^4')
    elif A == 3:
        # Fourier Basis (M = 8)
        p = np.hstack((np.zeros((N,1)), np.ones((N,1)), np.sin((np.pi)*X_star), np.cos((np.pi)*X_star), np.sin(2*(np.pi)*X_star), np.cos(2*(np.pi)*X_star), np.sin(3*(np.pi)*X_star), np.cos(3*(np.pi)*X_star), np.sin(4*(np.pi)*X_star), np.cos(4*(np.pi)*X_star), np.sin(5*(np.pi)*X_star), np.cos(5*(np.pi)*X_star), np.sin(6*(np.pi)*X_star), np.cos(6*(np.pi)*X_star), np.sin(7*(np.pi)*X_star), np.cos(7*(np.pi)*X_star), np.sin(8*(np.pi)*X_star), np.cos(8*(np.pi)*X_star)))
        p = p[:,1:10]
        l = ('1','sin(pi*x)','cos(pi*x)','sin(2*pi*x)','cos(2*pi*x)', 'sin(3*pi*x)','cos(3*pi*x)', 'sin(4*pi*x)','cos(4*pi*x)')
    elif A == 4:    
        # Legendre Basis
        p = np.hstack((np.ones((N,1)), X_star, 0.5*(3*(X_star**2) - 1), 0.5*(5*(X_star**3) - 3*X_star), 0.125*(35*(X_star**4) - 30*(X_star**2) + 3), 0.125*(63*(X_star**5) - 70*(X_star**3) + 15*X_star), 0.0625*(231*(X_star**6) - 315*(X_star**4) + 105*(X_star**2) - 5), 0.0625*(429*(X_star**7) - 693*(X_star**5) + 315*(X_star**3) - 35*X_star), (1/128)*(6435*(X_star**8) - 12012*(X_star**6) + 6930*(X_star**4) - 1260*(X_star**2) + 35)))
        p = p[:,0:5]
        l = ('L0 = 0','L1','L2','L3','L4')
    else:
        phi = 0
        print("Input a valid A correponding to the right basis")
    
    plt.figure(1, figsize=(12,9))
    plt.plot(X_star, p, linewidth=3.0)
    plt.legend(l)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    

    