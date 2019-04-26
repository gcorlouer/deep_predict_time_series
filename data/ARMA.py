#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:48:21 2019

@author: gc349
"""
#%% Import libraries
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
#%% Generate AR data 
#ARMA(1)
nsample=100000
arparams = np.array([.95])
maparams = np.array([0])
arparams = np.r_[1, -arparams]#step is zero is unweighted
maparams = np.r_[1, maparams]#step is zero is unweighted
y = arma_generate_sample(arparams, maparams, nsample)#mean zero variance 1
plt.figure()
plt.plot(y)
fname='ARMA.csv'
np.savetxt(fname,y)
