#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 07:43:01 2022

@author: aman
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import cv2
df=pd.read_csv("./datayt0.csv")
df = df.drop("frame", 1)
df.dropna(axis=0,how="any",inplace=True)
# self.df = df[df.state != "None"]
df.index = np.arange(0, len(df))
for i in [500,600,1500,1600]:    
    feature =   df.iloc[i:i+100, 0:132]
    feature = np.array([pd.to_numeric(feature.iloc[i], errors='coerce') for i in range(len(feature))])
    try:
        nose_x = feature[:,0].reshape((10,10))*200
        nose_x=nose_x.astype(np.uint8)
        nose_y = feature[:,1].reshape((10,10))*200
        nose_y=nose_y.astype(np.uint8)
        nose_v = feature[:,3].reshape((10,10))*100
        nose_v=nose_v.astype(np.uint8)
        image=cv2.merge([nose_v,nose_y,nose_x])
        if i<1400:
            name="squats"
        else :
            name="back lunges"
        plt.figure()
        plt.title(f"frame:{i},{name}")
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.axis("off")
        plt.imshow(nose_x,cmap="gray")
    except Exception as e:
        print(e)
        pass