# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:15:11 2023

@author: dylan
"""

import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('C:/Users/dylan/Documents/GitHub/PythonDetection/Haarcascades/haarcascade_frontalface_alt.xml')
eye_classifier = cv2.CascadeClassifier('C:/Users/dylan/Documents/GitHub/PythonDetection/Haarcascades/haarcascade_eye.xml')
plate_classifier = cv2.CascadeClassifier('C:/Users/dylan/Documents/GitHub/PythonDetection/Haarcascades/haarcascade_russian_plate_number.xml')
cat_classifier = cv2.CascadeClassifier('C:/Users/dylan/Documents/GitHub/PythonDetection/Haarcascades/haarcascade_frontalcatface_extended.xml')
lowerbody_classifier = cv2.CascadeClassifier('C:/Users/dylan/Documents/GitHub/PythonDetection/Haarcascades/haarcascade_lowerbody.xml')

img = cv2.imread('C:/Users/dylan/Documents/GitHub/PythonDetection/ImageDetection/russianplate.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.2, 3)#1.3=1.5 & < 1.1 5 =>5 & < 2

#when no faces detected, face classifier returns empty and tuple
if faces is ():
    print("no face found")

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h),(127, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detect(roi_gray)
    
    for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),(255, 255, 0), 2)
         cv2.imshow('img', img)
         cv2.waitKey(0)
        
cv2.destroyAllWindows()    