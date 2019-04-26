#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:09:53 2019

@author: gc349
"""
#%%
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys
#%%
def preparedata(file, target):
    """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
    df_data = pd.read_csv(file)
    df_y = df_data.copy(deep=True)[[target]]
    df_x = df_data.copy(deep=True)
    df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
    df_yshift[target]=df_yshift[target].fillna(0.)
    df_x[target] = df_yshift
    data_x = df_x.values.astype('float32').transpose()    
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    x, y = Variable(data_x), Variable(data_y)
    return x, y
#%% Train model over one epoch
def train(epoch, traindata, traintarget, modelname, optimizer,log_interval,epochs):
    """Trains model by performing one epoch and returns loss."""

    modelname.train()
    x, y = traindata[0:1], traintarget[0:1]
        
    optimizer.zero_grad()
    epochpercentage = (epoch/float(epochs))*100
    output = modelname(x)
    
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()

    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print('Epoch: {:  2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

    return loss
