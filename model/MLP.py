#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:16:40 2019

@author: gc349
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
#%%model MLP with arbitrary number of hidden layers
class MLP(torch.nn.Module):
    def __init__(self, h_sizes, out_size):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        output= self.out(x)
        return output