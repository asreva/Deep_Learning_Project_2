# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define loss function for the network
"""

# ----- Libraries ----- #
#Utils
from torch import set_grad_enabled
set_grad_enabled(False)

def MSE(value, target):
    #Compute the mean square error btw a given value tensor and a target tensor
    #Assume first dimension separate the different points (others may be used for dimension)
    return ((value - target)**2).sum()/value.shape[0]

def dMSE(value, target):
    return 2 * (value - target)