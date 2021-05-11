# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Plot a comparison btw results obtained by test.py
"""

# ----- Libraries ----- #
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import sys

# ----- Parameters ----- #
PATHS = ["ours_no_lr/","pytorch/"]
NAMES = ["Ours (without lr update)","Pytorch"]
COLORS = ["green","blue","red","black"]

fig0, ax0 = plt.subplots()
for i in range(len(NAMES)):
    if i > len(COLORS):
        sys.exit("ADD COLORS TO COLORS")
    # ----- Loading ----- #
    test_e = np.genfromtxt(PATHS[i]+'test_perf_e.txt', delimiter=' ')
    train_e = np.genfromtxt(PATHS[i]+'train_perf_e.txt', delimiter=' ')
    test = np.genfromtxt(PATHS[i]+'test_perf.txt', delimiter=' ')
    train = np.genfromtxt(PATHS[i]+'train_perf.txt', delimiter=' ')
    
    # ----- Stats ----- #
    test_e_m = np.mean(test_e,axis=0)
    test_e_s = np.std(test_e,axis=0)
    train_e_m = np.mean(train_e,axis=0)
    train_e_s = np.std(train_e,axis=0)
    test_m = np.mean(test)
    test_s = np.std(test)
    train_m = np.mean(train)
    train_s = np.std(train)
    
    # ----- Plot ----- #
    #Plot normal perf
    ax0.errorbar(i, test_m, yerr = test_s, capsize=10, label = "Test", color = "blue")
    ax0.errorbar(i, train_m, yerr = train_s, capsize=10, label = "Train", color = "red")
    
    #Plot the evolution on all the epochs for each parameter
    plt.figure(2)
    epochs_arr = np.arange(0, len(test_e_m))
    plt.errorbar(epochs_arr, train_e_m,yerr=train_e_s, label=NAMES[i]+" (train)",capsize=5, color = COLORS[i])
    plt.errorbar(epochs_arr, test_e_m,yerr=test_e_s, label=NAMES[i]+" (test)", linestyle="--",capsize=5, color = COLORS[i])


ax0.set_title("Error rate for different models, with std")
ax0.set_xlabel("Models")
ax0.set_ylabel("Error rate")
ax0.set_xticks(np.arange(len(NAMES)))
ax0.set_xticklabels(NAMES)
ax0.legend()

plt.figure(2)
plt.title("Evolution of the error rate along epochs, with std")
plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.legend()