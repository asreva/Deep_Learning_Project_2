# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define activation functions for the NN implementation
"""

# ----- Libraries ----- #
from modules import Module
from torch import set_grad_enabled, empty
set_grad_enabled(False)

# ----- Actiavtion definitions ----- #

#Tanh
class Tanh(Module):
    #Applies the Tanh activation layer to the input tensor
        
    def __init__(self):
        self.name = "Tanh Activation Layer"
        self.output = None 
        
    def forward(self, input):
        self.output = input.tanh()
        return self.output 

    def backward(self, gradwrtoutput):
        return (1-self.output.tanh().pow(2))*gradwrtoutput

#ReLU
class ReLU(Module):
    #Applies the ReLU activation layer to the input tensor
        
    def __init__(self):
        self.name = "ReLU Activation Layer"
        self.output = None

    def forward(self, input):
        input[input<0] = 0
        self.output = input
        return self.output

    def backward(self, gradwrtoutput):
        out = empty(size=(self.output.shape))
        out[self.output < 0] = 0
        out[self.output >= 0] = 1
        return out*gradwrtoutput
    
#Sigmoid
class Sigmoid(Module):
    #Applies the Sigmoid activation layer to the input tensor
        
    def __init__(self):
        self.name = "Sigmoid Activation Layer"
        self.output = None 
    
    def forward(self, input):
        self.output = input.sigmoid()
        return self.output 

    def backward(self, gradwrtoutput):
        return self.output.sigmoid()*(1-self.output.sigmoid()) * gradwrtoutput
