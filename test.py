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
N_EPOCHS = 1000
eta = 1e-2
# ----- Functions ----- #

# ----- Main ----- #
train_points, train_labels, test_points, test_labels = datasets.generate_circle_dataset(N)

seq = modules.Sequential()

seq.append(layers.FCLayer(2, 25, True))
seq.append(act.Tanh())
seq.append(layers.FCLayer(25, 25, True))
seq.append(act.Tanh())
seq.append(layers.FCLayer(25, 1, True))
seq.append(act.Tanh())

print(seq.param())

seq.names()
for epoch in range(N_EPOCHS):
    err_train = 0
    err_test = 0
    for i in range(N):
        pred_train = seq.forward(train_points[i, :])

        seq.backward(loss.dMSE(pred_train.view(1, -1), train_labels[i].view(1, -1)))
        seq.grad_step(eta)
        if (train_labels[i] and pred_train <= 0.5) or (not train_labels[i] and pred_train >= 0.5):
            err_train += 1

        pred_test = seq.forward(test_points[i, :])
        if (test_labels[i] and pred_test <= 0.5) or (not test_labels[i] and pred_test >= 0.5):
            err_test += 1

    print("train epoch {} err = {:.2%}".format(epoch, err_train / N))
    print("test epoch {} err = {:.2%}".format(epoch, err_test / N))
    # for layer in seq.layers:
    #     layer.grad_step(eta)


