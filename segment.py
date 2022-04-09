#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:28:54 2022

@author: aman
"""
import torch
import cv2
import mediapipe as mp
from model import Conv2Net,key,Net1,Net
from collections import deque
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_file")
parser.add_argument("--output_file")

args = parser.parse_args()
def get_key(val):
    for k, value in key.items():
         if val == value:
             return k
    return "key doesn't exist"
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# file="media/video0_0.mp4"
output_file=args.output_file
if str.isdigit(args.input_file):
    file=int(args.input_file)
else:
    file=args.input_file
cap=cv2.VideoCapture(file)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
if cap.isOpened(): 
    # get vcap property 
    width  = int(cap.get(3))   # float `width`
    height = int(cap.get(4)) # float `height`
out = cv2.VideoWriter(output_file, fourcc, 20.0, (width,height))

net=Net1().double()
net.load_state_dict(torch.load("net.pt"))
net.eval()

WINDOW=100
d=deque([],WINDOW)
row=[]
Ex="_"

with mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break
        height, width, _ = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        row=[]
        try:
            for j in range(33):
                x = results.pose_landmarks.landmark[j].x
                y = results.pose_landmarks.landmark[j].y
                z = results.pose_landmarks.landmark[j].z
                v = results.pose_landmarks.landmark[j].visibility
                row.append(x)
                row.append(y)
                row.append(z)
                row.append(v)
            d.append(row)
            if(len(d)<WINDOW):
                Ex="_"
            else:
                feature=torch.tensor([d]).permute((0,2,1)).double()
                output=net(feature)
                _, predicted = torch.max(output.data, 1)
                Ex=get_key(predicted)
        except Exception as e:
            # print(e)
            Ex="__"
            pass
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    # str(round(angle1)),
                    str(Ex),
                    (100, 100), font,
                    1, (0, 0, 255),
                    2, cv2.LINE_AA)
        out.write(image)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
out.release()