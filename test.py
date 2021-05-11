# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Test the manual framework on a simple example
"""

# ----- Libraries ----- #
#Utils
from torch import set_grad_enabled
set_grad_enabled(False)
#Ours
import datasets 
import act_func as act
import loss
import layers
import modules
from lr_adapter import LearningRateAdapter

# ----- Parameters ----- #
N = 1000                #nb of dats in both train and test dataset
N_EPOCHS = 30           #nb of epoch for the train
eta0 = 1e-1             #initial learning rate
N_ITER = 10             #nb of iter to compute mean and std
AUTO_LR_REDUCE = True   #enable auto lr updater
IMPROVEMENT_TH = 0.01   #min improvement th to increase counter
LR_RED = 0.5            #lr update factor 
MAX_LR_CNT = 2          #value that the counter needs to reach before lr update
BOOL_SAVE = True        #to save or not the data
VERBOSE = True          #display information


# ----- Log tables ----- #
train_perf = []     #final perf train of each iter
test_perf = []      #fian perf test of each iter
train_perf_i_e = [] #perf train of each iter along epochs
test_perf_i_e = []  #perf test of each iter along epochs

# ----- Main ----- #
#repeat for stat info
for iter in range(N_ITER):
    eta = eta0  # reset learning rate
    if VERBOSE:
        print("\nIter: "+str(iter)+"\n")
    
    #Get data
    train_points, train_labels, test_points, test_labels = \
                                                datasets.generate_circle_dataset(N)
    train_perf_e = [] #perf train along epochs
    test_perf_e = []  #perf test along epochs
    
    #Define the network
    seq = modules.Sequential()
    
    seq.append(layers.FCLayer(2, 25, True))
    seq.append(act.Tanh())
    seq.append(layers.FCLayer(25, 25, True))
    seq.append(act.Tanh())
    seq.append(layers.FCLayer(25, 1, True))
    seq.append(act.Sigmoid())
    
    if VERBOSE:
        seq.names()
    
    #Define the lr adapter if needed
    if AUTO_LR_REDUCE:
        lr_adapter = LearningRateAdapter(eta0, IMPROVEMENT_TH, LR_RED, MAX_LR_CNT)
    
    #Train the network
    for epoch in range(N_EPOCHS): #for each epoch
        err_train = 0
        err_test = 0
        for i in range(N): #for each point
            #predict the output and backward the loss
            pred_train = seq.forward(train_points[i, :])
            seq.backward(loss.dMSE(pred_train.view(1, -1), \
                                   train_labels[i].view(1, -1)))
            seq.grad_step(eta)
            
            #check performance
            if (train_labels[i] and pred_train <= 0.5) \
                                    or (not train_labels[i] and pred_train >= 0.5):
                err_train += 1
            pred_test = seq.forward(test_points[i, :])
            if (test_labels[i] and pred_test <= 0.5) \
                                    or (not test_labels[i] and pred_test >= 0.5):
                err_test += 1
        if VERBOSE:
            print("train epoch {} err = {:.2%}".format(epoch, err_train / N))
            print("test epoch {} err = {:.2%}".format(epoch, err_test / N))

        if epoch > 1 and AUTO_LR_REDUCE: #decrease eta if stagnation
            improvement = (train_perf_e[epoch-1]-err_train/N)
            if VERBOSE:
                print("train improvement  = {:.2%}".format(improvement))
            eta = lr_adapter.step(improvement)

        train_perf_e.append(err_train / N)
        test_perf_e.append(err_test / N)
        
    #Test the network
    #on test
    err_test = 0  
    for i in range(N): #for each point
        #predict the output
        pred_test = seq.forward(test_points[i, :])
        #add errors
        if (test_labels[i] and pred_test <= 0.5) \
                                or (not test_labels[i] and pred_test >= 0.5):
            err_test += 1
    
    #on train
    err_train = 0     
    for i in range(N): #for each point
        #predict the output
        pred_train = seq.forward(train_points[i, :])
        #add errors
        if (train_labels[i] and pred_train <= 0.5) \
                                or (not train_labels[i] and pred_train >= 0.5):
            err_train += 1
    
    #save and print
    train_perf.append(err_train / N)
    test_perf.append(err_test / N)
    train_perf_i_e.append(train_perf_e)
    test_perf_i_e.append(test_perf_e)
    if VERBOSE:
        print("\nFinal train {} err = {:.2%}".format(epoch, err_train / N))
        print("Final test {} err = {:.2%}".format(epoch, err_test / N))
    
#save in file
if BOOL_SAVE:
    with open('train_perf.txt', 'w+') as f:
        for val in train_perf:
            f.write(str(val)+" ")
        
    with open('test_perf.txt', 'w+') as f:
        for val in test_perf:
            f.write(str(val)+" ")
            
    with open('train_perf_e.txt', 'w+') as f:
        for l in train_perf_i_e:
            for val in l:
                f.write(str(val)+" ")
            f.write("\n")
        
    with open('test_perf_e.txt', 'w+') as f:
        for l in test_perf_i_e:
            for val in l:
                f.write(str(val)+" ")
            f.write("\n")
