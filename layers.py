# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Define classic NN layers
"""

# ----- Libraries ----- #
from math import sqrt
from torch import empty
from modules import Module
from torch import set_grad_enabled
set_grad_enabled(False)


# ----- Layers definitions and functions ----- #

#Layer init
def layer_init(X, init="Normal", param1=None, param2=None, input_w=0, output_w=0):
    #initialise the layer following the instructions or default choices
    #   - X:        the tensor to initialise (maybe W or b)
    #   - init:     the type of init (Normal, Uniform or Xavier)
    #   - param1/2  the parameters of the init (for normal it is mean and std, 
    #               for uniform it is min and max value, no param for xavier)
    #   - input_w:      size of the input (nb input connections)
    #   - output_w:     size of the output (nb output connections)
    
    if init=="Normal":
        if param1==None:
            mean=0
        if param2==None:
            std=1
        X = X.normal_(mean=mean, std=std)  
    elif init=="Uniform":
        if param1==None:
            v_min=-sqrt(input_w)
        if param2==None:
            v_max=sqrt(input_w)
        X = X.uniform_(v_min, v_max)  
    elif init=="Xavier":
        v_min=-sqrt(6)/sqrt(output_w+input_w)
        v_max=sqrt(6)/sqrt(output_w+input_w)
        X = X.uniform_(v_min, v_max)
    else:
        print("This initialiation is unknown")
        
    return X

#FC Layer
class FCLayer(Module):
    #Generates a Fully Connected Layer
    #Parameters:
    #   - input_w:      size of the input (nb input connections)
    #   - output_w:     size of the output (nb output connections)
    #   - bias_bool:    boolean, if a bias is added to the output of the FC
    #   - init:     the type of init (Normal, Uniform or Xavier)
    #   - param1/2  the parameters of the init (for normal it is mean and std, 
    #               for uniform it is min and max value, no param for xavier)
    
    def __init__(self, input_w, output_w, bias_bool, init="Normal", param1=None, param2=None):
        self.name = "FC Layer with input "+str(input_w)+", output "+str(output_w)+" and bias set to "+str(bias_bool)
        self.bias_bool = bias_bool  
        self.input = empty(size=(input_w,1))
        self.output = empty(size=(output_w,1))
        self.dl_dw = empty(size=(output_w, input_w)).zero_()
        
        self.W = empty(size=(output_w, input_w)) # weights of the layer
        self.W = layer_init(self.W, init, param1, param2, input_w, output_w)
        
        if bias_bool:
            self.b = empty(output_w)
            self.b = layer_init(self.b, init, param1, param2, input_w, output_w)
            self.dl_db = empty(output_w).zero_()

    def forward(self, input):
        self.input = input
        self.output = self.W.matmul(input)
        if self.bias_bool:
            self.output += self.b
        return self.output


    def backward(self, gradwrtoutput):
        self.dl_dw.add_(gradwrtoutput.view(-1, 1).matmul(self.input.view(1, -1)))
        if self.bias_bool:
            self.dl_db.add_(gradwrtoutput.squeeze())
        return self.W.t().matmul(gradwrtoutput).squeeze()

    def grad_step(self,eta):
        self.W -= eta * self.dl_dw
        self.dl_dw.zero_()
        if self.bias_bool:
            self.b -= eta * self.dl_db
            self.dl_db.zero_()


    def param(self):
        params = []
        params.append((self.W,self.dl_dw))
        if self.bias:
            params.append((self.b,self.dl_db))
            
        return params
