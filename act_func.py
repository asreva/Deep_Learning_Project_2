# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define activation functions for the NN implementation
"""

# ----- Libraries ----- #
from modules import Module
from torch import set_grad_enabled
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
        return 4 * (self.output.exp() + self.output.mul(-1).exp()).pow(-2) * gradwrtoutput

#ReLU
class ReLU(Module):
    #Applies the ReLU activation layer to the input tensor
        
    def __init__(self):
        self.name = "ReLU Activation Layer"
    
    def forward(self, input):
        input[input<0] = 0 
        return input

    def backward(self, gradwrtoutput):
        gradwrtoutput[gradwrtoutput < 0] = 0
        gradwrtoutput[gradwrtoutput >= 0] = 1
        return gradwrtoutput
    
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
