#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:10:20 2022

@author: aman
"""
import torch
import torch.nn as nn
key={"None":0,"squats":1,"standing":2,"pushup":3}
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main=nn.Sequential(
        nn.Conv1d(in_channels=1,out_channels=1,kernel_size=4,stride=1),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        # nn.Conv1d(5,5,33,33),
        # nn.BatchNorm1d(5),
        # nn.ReLU(),
        # nn.Conv1d(10,100,3,1),
        # nn.BatchNorm1d(100),
        # nn.ReLU(),
        nn.Flatten(),
        nn.Linear((13200-4+1)*1,1000),
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Linear(1000,4),
        #nn.Softmax()
        )
    def forward(self,x):
        return self.main(x)
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.main=nn.Sequential(
        nn.Conv1d(132,200,4,1),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        
        nn.Conv1d(200,200,3,1),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Flatten(),
        
        nn.Linear(95*200,4),
        #nn.Softmax()
        )
    def forward(self,x):
        return self.main(x)
class Conv2Net(nn.Module):
    def __init__(self):
        super(Conv2Net, self).__init__()
        self.main=nn.Sequential(
        nn.Conv2d(132,200,3,padding=1),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        # nn.Conv2d(200,200,3,padding=1),
        # nn.BatchNorm2d(200),
        # nn.ReLU(),
        nn.Flatten(),
        nn.Linear(20000,4),
        #nn.Softmax()
        )
    def forward(self,x):
        return self.main(x)
