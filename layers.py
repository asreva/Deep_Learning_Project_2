# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define classic NN layers
"""

# ----- Libraries ----- #
from torch import empty
from modules import Module

# ----- Layers definitions ----- #

#FC Layer
class FCLayer(Module):
    #Generates a Fully Connected Layer
    #Parameters:
    #   - input_w:      size of the input (nb input connections)
    #   - output_w:     size of the output (nb output connections)
    #   - bias:         boolean, if a bias is added to the output of the FC
    
    def __init__(self, input_w, output_w, bias):
        self.name = "FC Layer with input "+str(input_w)+", output "+str(output_w)+" and bias set to "+str(bias)
        self.bias = bias
        self.W = empty(size=(output_w, input_w)).normal_()      #weights of the layer
        self.input = empty(size=(input_w,1))
        self.output = empty(size=(output_w,1))
        self.dl_dw = empty(size=(output_w, input_w))
        if bias:
            self.b = empty(output_w).normal_()                  #bias of the layer
            self.dl_db = empty(output_w)

    def forward(self, input):
        self.input = input
        self.output = self.W.matmul(input)
        if self.bias:
            self.output += self.b
        return self.output


    def backward(self, gradwrtoutput):
        self.dl_dw.add_(gradwrtoutput.view(-1, 1).matmul(self.input.view(1, -1)))
        if self.bias:
            self.dl_db.add_(gradwrtoutput)
        return self.W.t().matmul(gradwrtoutput).squeeze()

    def param(self):
        params = []
        params.append((self.W,self.W))        #TODO 2nd is supposed to be the gradient
        if self.bias:
            params.append((self.b,self.b))    #TODO 2nd is supposed to be the gradient
            
        return params
    
    #TODO: Add backward