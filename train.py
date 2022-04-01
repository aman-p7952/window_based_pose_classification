#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:49:09 2022

@author: aman
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
WINDOW=100
key={"None":0,"squat":1,"back lunge":2}
class PoseDataset(Dataset):
    def __init__(self, file):
        self.file = file
        df = pd.read_csv(file)
        self.df = df.drop("frame", 1)
        self.df.dropna(axis=0,how="any",inplace=True)
        # self.df = df[df.state != "None"]
        self.df.index = np.arange(0, len(self.df))
    def __len__(self):
        return len(self.df)-WINDOW+1
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.df.iloc[idx:idx+100, 0:132]
        feature = np.array([pd.to_numeric(feature.iloc[i], errors='coerce') for i in range(len(feature))])
        #feature=feature.reshape((1,-1))
        label = self.df.iloc[idx:idx+100,132]
        label = list(label)
        unique , count =np.unique(label,return_counts=True)
        index=np.argmax(count)
        label=unique[index]
        label=torch.tensor(key[label]).long()
        sample = {"feature": torch.tensor(feature),"label":label}
        return sample
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main=nn.Sequential(
        nn.Conv1d(1,1, 10*132),
        nn.BatchNorm1d(1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100*132-10*132+1,3),
        #nn.Softmax()
        )
    def forward(self,x):
        return self.main(x)
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.main=nn.Sequential(
        nn.Conv1d(100,100,5),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Conv1d(100,100,5),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Flatten(),
        
        nn.Linear((132-10+2)*100,3),
        #nn.Softmax()
        )
    def forward(self,x):
        return self.main(x)
dataset=PoseDataset("datayt0.csv")
dataloader=DataLoader(dataset,2,shuffle=True)
dataset_test=PoseDataset("data1.csv")
dataloader_test=DataLoader(dataset_test,2,shuffle=True)
net=Net1().double()
n_epochs=10
optimizer = optim.Adam(net.parameters(), lr=0.001)
critarian = nn.CrossEntropyLoss()
train_history = []
test_history = []
losses = []
for epoch in range(n_epochs):
    for i,sample in enumerate(dataloader):
        # print(sample["feature"].shape)
        # print(sample["label"].shape)
        optimizer.zero_grad()
        output=net(sample["feature"])
        loss=critarian(output,sample["label"])
        loss.backward()
        optimizer.step()
        # print(loss.detach().numpy())
        losses.append((loss.detach().numpy()))
    with torch.no_grad():
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        for data in dataloader:
            features = data["feature"].double()
            label = data["label"].long()
            outputs = net(features)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            #_, label = torch.max(label.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
        
        train_acc = correct_train/total_train
        train_history.append(train_acc)
        for data in dataloader_test:
            features = data["feature"].double()
            label = data["label"].long()
            outputs = net(features)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            #_, label = torch.max(label.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum().item()
        test_acc = correct_test/total_test
        test_history.append(test_acc)
    loss_mean=np.array(losses).mean()
    print("epoch: %d mean_loss %f train_accracy %f test_accuracy %f" %
          (epoch+1, loss_mean, train_acc,test_acc))
plt.plot(train_history)
plt.title("trainning history")
plt.xlabel("epochs")
plt.ylabel("accuracy")
torch.save(net.state_dict(), "net.pt")