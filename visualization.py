#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 08:52:35 2022

@author: aman
"""

import pandas as pd
import numpy as np
import cv2
import os

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
df=pd.read_csv("./datayt0.csv")
df = df.drop("frame", 1)
df.dropna(axis=0,how="any",inplace=True)
# self.df = df[df.state != "None"]
df.index = np.arange(0, len(df))
file="./video0_0.mp4"
cap = cv2.VideoCapture(file)
i=0
if cap.isOpened(): 
    # get vcap property 
    width  = int(cap.get(3))   # float `width`
    height = int(cap.get(4)) # float `height`
out = cv2.VideoWriter('./labeled0.mp4', fourcc, 20.0, (width,height)) 
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        # continue
        break
    # height, width, _ = image.shape
    if i<100:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    # str(round(angle1)),
                    str("-"),
                    (100, 100), font,
                    0.7, (0, 0, 255),
                    1, cv2.LINE_AA)
        # cv2.imshow('MediaPipe Pose', image)
        if not os.path.isdir('./images/all1'):
            os.mkdir('./images/all1')
        filename = "./images/all1/all"+str(i)+".jpg"
        cv2.imwrite(filename, image)
        out.write(image)
    else:
        try:
            label = df.iloc[i-100:i,132]
            label = list(label)
            unique , count =np.unique(label,return_counts=True)
            index=np.argmax(count)
            label=unique[index]
        except Exception as e:
            label="None"
    
        # image = cv2.flip(image, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    # str(round(angle1)),
                    str(label),
                    (100, 100), font,
                    0.7, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', image)
        if not os.path.isdir('./images/all1'):
            os.mkdir('./images/all1')
        filename = "./images/all1/all"+str(i)+".jpg"
        cv2.imwrite(filename, image)
        out.write(image)
    i+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()