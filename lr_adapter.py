# -*- coding: utf-8 -*-
"""
@authors:   Isaac Dinis & Ivan-Daniel Sievering
@aim:       Implement an automatic learning rate adapter
"""

# ----- Libraries ----- #
#Utils
from torch import set_grad_enabled
set_grad_enabled(False)

# ----- Learning Rate adapter ----- #
class LearningRateAdapter(object):
    #Learning Rate Adapter instance.
    #Each time the improvement is smaller than th, a counter is incremented,
    #when this counter is higher er equal to max_cnt_red, the learning rate,
    #initialised at init_lr is multiplied by lr_red and returned
    
    def __init__(self, init_lr, th, lr_red, max_cnt_red):
        self.lr = init_lr
        self.th = th
        self.lr_red = lr_red
        self.max_cnt_red = max_cnt_red
        self.cnt = 0

    def step(self, improvement):
        #increment counter if big improvement
        if improvement < self.th:  
            self.cnt += 1
        #if the counter exceeded the max value update the lr and reset the cnt
        if self.cnt >= self.max_cnt_red: 
            self.cnt = 0
            self.lr = self.lr * self.lr_red
        return self.lr
