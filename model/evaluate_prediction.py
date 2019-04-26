#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:32:15 2019

@author: gc349
"""
import optim
import argparse
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
#import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys
#%% A revoir
def evaluate_prediction(target, cuda, epochs, kernel_size, layers, 
               loginterval, lr, optimizername, seed, dilation_c, split, file):
    """Runs first part of model to predict one time series and evaluate its accuracy (MASE)."""
    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    X, Y = optim.preparedata(file, target)
    X = X.unsqueeze(0).contiguous()
    Y = Y.unsqueeze(2).contiguous()

    timesteps = X.size()[2]
    if timesteps!=Y.size()[1]:
        print("WARNING: Time series do not have the same length.")
    X_train = X[:,:,:int(split*timesteps)]
    Y_train = Y[:,:int(split*timesteps),:]
    X_test = X[:,:,int(split*timesteps):]
    Y_test = Y[:,int(split*timesteps):,:]

    input_channels = X_train.size()[1]
    targetidx = pd.read_csv(file).columns.get_loc(target)
        model = MLP(targetidx, input_channels, levels)

    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    for ep in range(1, epochs+1):
        scores, realloss = TCDF.train(ep, X_train, Y_train, model, optimizer,loginterval,epochs)
    realloss = realloss.cpu().data.item()

    model.eval()
    output = model(X_test)
    prediction=output.cpu().detach().numpy()[0,:,0]
    T = output.size()[1]
    total_e = 0.
    for t in range(T):
        real = Y_test[:,t,:]
        predicted = output[:,t,:]
        e = abs(real - predicted)
        total_e+=e
    total_e = total_e.cpu().data.item()
    total = 0.
    for t in range(1,T):
        temp = abs(Y_test[:,t,:] - Y_test[:,t-1,:])
        total+=temp
    denom = (T/float(T-1))*total
    denom = denom.cpu().data.item()

    if denom!=0.:
        MASE = total_e/float(denom)
    else:
        MASE = 0.
    
return MASE, prediction