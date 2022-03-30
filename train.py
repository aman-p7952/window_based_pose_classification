#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:49:09 2022

@author: aman
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
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
        feature=feature.reshape((-1))
        label = self.df.iloc[idx:idx+100,132]
        label = list(label)
        unique , count =np.unique(label,return_counts=True)
        index=np.argmax(count)
        label=unique[index]
        sample = {"feature": torch.tensor(feature),"label":torch.tensor(key[label])}
        return sample
    
dataset=PoseDataset("datayt0.csv")
datalosder=DataLoader(dataset,2,shuffle=False)

for i,sample in enumerate(datalosder):
    print(sample["feature"].shape)
    print(sample["label"])