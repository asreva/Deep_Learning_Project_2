# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Test the manual framework on a simple example
"""

# ----- Libraries ----- #
#Utils
import math
from torch import empty
from torch import set_grad_enabled
set_grad_enabled(False)
#Ours
import datasets 
import act_func as act
import loss
import layers
import modules



# ----- Parameters ----- #
N = 1000           #nb of dats in both train and test dataset

# ----- Functions ----- #

# ----- Main ----- #
train_points, train_labels, test_points, test_labels = datasets.generate_circle_dataset(N)

seq = modules.Sequential()

seq.append(layers.FCLayer(2, 25, True))
seq.append(act.Tanh())
seq.append(layers.FCLayer(25, 25, True))
seq.append(act.ReLU())
seq.append(layers.FCLayer(25, 1, False))


print(seq.param())

seq.names()

print(seq.forward(train_points[0,:]))

seq.backward(loss.dMSE(test_labels[0].view(1,-1), test_labels[1].view(1,-1)))




