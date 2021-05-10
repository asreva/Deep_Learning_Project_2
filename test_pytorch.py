# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Test the manual framework on a simple example
"""

# ----- Libraries ----- #
#Ours
import datasets 
#Utils
from torch import nn
import torch.optim as optim
from torch import set_grad_enabled
set_grad_enabled(True)

# ----- Parameters ----- #
N = 1000            #nb of dats in both train and test dataset
N_EPOCHS = 30      #nb of epoch for the train
eta = 1e-1          #learning rate
N_ITER = 10         #nb of iter to compute mean and std
BOOL_SAVE = True    #to save or not the data

# ----- Log tables ----- #
train_perf = []     #final perf train of each iter
test_perf = []      #fian perf test of each iter
train_perf_i_e = [] #perf train of each iter along epochs
test_perf_i_e = []  #perf test of each iter along epochs

# ----- Functions ----- #
def init_weights(m):    
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0, std=1)  
        m.bias.data.normal_(mean=0, std=1)  

# ----- Main ----- #
#repeat for stat info
for iter in range(N_ITER):
    print("\nIter: "+str(iter)+"\n")
    
    #Get data
    train_points, train_labels, test_points, test_labels = datasets.generate_circle_dataset(N)
    train_perf_e = [] #perf train along epochs
    test_perf_e = []  #perf test along epochs
    
    #Define the network
    seq = nn.Sequential(
        nn.Linear(2,25),
        nn.Tanh(),
        nn.Linear(25,25),
        nn.Tanh(),
        nn.Linear(25,1),
        nn.Sigmoid()
    )
    seq.apply(init_weights)
    
    optimizer = optim.SGD(seq.parameters(), lr=eta)
    criterion = nn.MSELoss()
    
    #Train the network
    for epoch in range(N_EPOCHS): #for each epoch
        err_train = 0
        err_test = 0
        for i in range(N): #for each point
            #predict the output and backward the loss
            pred_train = seq(train_points[i, :])
            loss = criterion(pred_train, train_labels[i].view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #check performance
            if (train_labels[i] and pred_train <= 0.5) or (not train_labels[i] and pred_train >= 0.5):
                err_train += 1
            pred_test = seq(test_points[i, :])
            if (test_labels[i] and pred_test <= 0.5) or (not test_labels[i] and pred_test >= 0.5):
                err_test += 1
                
        print("train epoch {} err = {:.2%}".format(epoch, err_train / N))
        print("test epoch {} err = {:.2%}".format(epoch, err_test / N))
        train_perf_e.append(err_train / N)
        test_perf_e.append(err_test / N)
        
    #Test the network
    #on test
    err_test = 0  
    for i in range(N): #for each point
        #predict the output
        pred_test = seq(test_points[i, :])
        #add errors
        if (test_labels[i] and pred_test <= 0.5) or (not test_labels[i] and pred_test >= 0.5):
            err_test += 1
    
    #on train
    err_train = 0     
    for i in range(N): #for each point
        #predict the output
        pred_train = seq(train_points[i, :])
        #add errors
        if (train_labels[i] and pred_train <= 0.5) or (not train_labels[i] and pred_train >= 0.5):
            err_train += 1
    
    #save and print
    train_perf.append(err_train / N)
    test_perf.append(err_test / N)
    train_perf_i_e.append(train_perf_e)
    test_perf_i_e.append(test_perf_e)
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
