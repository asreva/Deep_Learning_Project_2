# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Plot the results obtained by test.py
"""

# ----- Libraries ----- #
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# ----- Constants ----- #
PATH = "ours_no_lr"+"/"

# ----- Loading ----- #
test_e = np.genfromtxt(PATH+'test_perf_e.txt', delimiter=' ')
train_e = np.genfromtxt(PATH+'train_perf_e.txt', delimiter=' ')
test = np.genfromtxt(PATH+'test_perf.txt', delimiter=' ')
train = np.genfromtxt(PATH+'train_perf.txt', delimiter=' ')

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
print("Mean train perf is "+str(train_m*100)+"% with std "+str(train_s*100)+"%")
print("Mean test perf is "+str(test_m*100)+"% with std "+str(test_s*100)+"%")

#Plot the evolution on all the epochs for each parameter
epochs_arr = np.arange(0, len(test_e_m))
plt.errorbar(epochs_arr, train_e_m,yerr=test_e_s, label="Train",capsize=5)
plt.errorbar(epochs_arr, test_e_m,yerr=test_e_s, label="Test", linestyle="--",capsize=5)
plt.title("Evolution of the error rate along epochs, with std")
plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.legend()