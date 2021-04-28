# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define loss function for the network
"""

def MSE(value, target):
    #Compute the mean square error btw a given value tensor and a target tensor
    #Assume first dimension separate the different points (others may be used for dimension)
    
    return ((value - target)**2).sum()/value.shape[0]