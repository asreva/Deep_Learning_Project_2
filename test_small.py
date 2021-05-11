# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Test the manual framework on a very simple example (the one in the report)
"""

# ----- Libraries ----- #
import datasets, loss, layers, modules, act_func

# ----- Parameters ----- #
N = 1000           #nb of datas in both train and test dataset
N_EPOCHS = 30      #nb of epoch for the train
eta = 1e-1         #learning rate

# ----- Get the data ----- #
train_points, train_labels, test_points, test_labels = datasets.generate_circle_dataset(N)

# ----- Define the network ----- #
seq = modules.Sequential()                  #create the sequential
seq.append(layers.FCLayer(2, 25, True))     #add an FC layer with bias
seq.append(act_func.Tanh())                 #add a Tanh layer
seq.append(layers.FCLayer(25, 25, True))
seq.append(act_func.Tanh())
seq.append(layers.FCLayer(25, 1, True))
seq.append(act_func.Sigmoid())              #add a Sigmoid layer

# ----- Train the network ----- #
for epoch in range(N_EPOCHS): #for each epoch
    for i in range(N):        #for each point
        #predict the output and backward the loss
        pred_train = seq.forward(train_points[i, :])
        seq.backward(loss.dMSE(pred_train.view(1, -1), train_labels[i].view(1, -1)))
        seq.grad_step(eta)
    
# ----- Test the network ----- #
err_test = 0  
for i in range(N): #for each point
    #predict the output
    pred_test = seq.forward(test_points[i, :])
    #add errors
    if (test_labels[i] and pred_test <= 0.5) or (not test_labels[i] and pred_test >= 0.5):
        err_test += 1
print("The error percentage is "+str(err_test*100/N)+"%")

