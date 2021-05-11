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
    return ((value - target)**2).sum()/value.shape[0]

def dMSE(value, target):
    #Compute derivative of the mean square error btw a given tensor and a target tensor
    return 2 * (value - target)/value.shape[0]