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
        self.output = 0 #TODO maybe add dimension
    def forward(self, input):
        self.output = input.tanh()
        return input.tanh() 

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
