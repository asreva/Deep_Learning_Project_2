# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define the common strucutre for the different element of NN
"""

# ----- Libraries ----- #
#Utils
from torch import set_grad_enabled
set_grad_enabled(False)

# ----- Classes definition ----- #

#Basic Module
class Module(object):
    #parent class for NN elements
    
    def __init__(self):
        self.name = "Basic Module"
    
    def forward(self, input):
        raise NotImplementedError
        
    def backward(self, gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []

    def grad_step(self, eta):
        return

#Sequential
class Sequential(Module):
    #Sequential class to create sequential networks
    
    def __init__(self):
        self.name = "Sequential Structure"
        self.layers = []
        
    def append(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            input=layer.forward(input)
        return input

    def backward(self, gradwrtoutput):
        for layer in self.layers[::-1]:
            gradwrtoutput = layer.backward(gradwrtoutput)
        return input

    def grad_step(self, eta):
        for layer in self.layers:
            layer.grad_step(eta)

    def param(self):
        params = []
        for layer in self.layers:
            params.append(layer.param())
        return params
    
    def names(self):
        print(self.name+" with "+str(len(self.layers))+" layers:")
        for layer in self.layers:
            print(layer.name)
        print("\n")