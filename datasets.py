# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Generate datasets for the NN to learn
"""

# ----- Libraries ----- #
import math
from torch import empty
from torch import set_grad_enabled
set_grad_enabled(False)

# ----- Constants ----- #
R = 1/math.sqrt(2*math.pi)      #radius of the disk
X_OFF = 0.5                     #disk X offset
Y_OFF = 0.5                     #disk Y offset

# ----- Functions ----- #
def generate_circle_dataset(N=1000):
    #generates a train and a test dataset with N points generated uniformly btw 0 and 1
    #points outside of the disk centered at [0.5 0.5] with a radius 1/sqrt(2*pi)
    #are classified as 0, the points inside (or on the boundary) as 1
    
    #Create points
    test_points = empty(N, 2).uniform_()
    train_points = empty(N, 2).uniform_()
    
    #Label the points
    test_labels = (test_points[:,0]-X_OFF)**2+(test_points[:,1]-Y_OFF)**2 <= R**2
    train_labels = (train_points[:,0]-X_OFF)**2+(train_points[:,1]-Y_OFF)**2 <= R**2
    
    #Conver to float (needed for MSE computation)
    test_labels = test_labels.type(test_points.dtype)
    train_labels = train_labels.type(train_points.dtype)
    
    return train_points, train_labels, test_points, test_labels





