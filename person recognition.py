# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2

body_classifier = cv2.CascadeClassifier('C:/Users/A00294976/Documents/New folder/Harrcascade/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('C:/Users/A00294976/Downloads/vid1.mp4')

while cap.isOpened():
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray, 1.1, 5)
    
    for(x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('person detection', frame)
    
    if cv2.waitKey(5) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()
