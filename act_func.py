# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define activation functions for the NN implementation
"""

# ----- Libraries ----- #
from modules import Module

# ----- Actiavtion definitions ----- #

#Tanh
class Tanh(Module):
    #Applies the Tanh activation layer to the input tensor
        
    def __init__(self):
        self.name = "Tanh Activation Layer"
    
    def forward(self, input):
        return input.tanh() 
    
    #TODO: Add backward

#ReLU
class ReLU(Module):
    #Applies the ReLU activation layer to the input tensor
        
    def __init__(self):
        self.name = "ReLU Activation Layer"
    
    def forward(self, input):
        input[input<0] = 0 
        return input
    
    #TODO: Add backward