import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from part1 import objpoints,imgpoints,objp

images = glob.glob('camera_cal/calib*.jpg')

i=0
plt.figure(figsize=(12,12))
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        i+=1
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.subplot(5, 6, i)
        plt.axis('off')
        plt.imshow(img)

