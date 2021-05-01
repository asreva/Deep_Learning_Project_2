# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define classic NN layers
"""

# ----- Libraries ----- #
from torch import empty
from modules import Module
import math
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
        #self.W = empty(size=(output_w, input_w)).uniform_(-math.sqrt(input_w),math.sqrt(input_w))      #weights of the layer
        self.W = empty(size=(output_w, input_w)).normal_(mean = 0, std = 1)  # weights of the layer
        self.input = empty(size=(input_w,1))
        self.output = empty(size=(output_w,1))
        self.dl_dw = empty(size=(output_w, input_w)).zero_()
        if bias:
            #self.b = empty(output_w).uniform_(-math.sqrt(input_w),math.sqrt(input_w))                   #bias of the layer
            self.b = empty(output_w).normal_(mean = 0, std = 1)
            self.dl_db = empty(output_w).zero_()

    def forward(self, input):
        self.input = input
        self.output = self.W.matmul(input)
        if self.bias:
            self.output += self.b
        return self.output


    def backward(self, gradwrtoutput):
        self.dl_dw.add_(gradwrtoutput.view(-1, 1).matmul(self.input.view(1, -1)))
        if self.bias:
            self.dl_db.add_(gradwrtoutput.squeeze())
        return self.W.t().matmul(gradwrtoutput).squeeze()

    def grad_step(self,eta):
        self.W -= eta * self.dl_dw
        self.dl_dw.zero_()
        if self.bias:
            self.b -= eta * self.dl_db
            self.dl_db.zero_()


    def param(self):
        params = []
        params.append((self.W,self.dl_dw))
        if self.bias:
            params.append((self.b,self.dl_db))
            
        return params
