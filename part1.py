import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

print ("Object Points As Follows",objp)

objpoints = [] 
imgpoints = [] 


images = glob.glob('camera_cal/calib*.jpg')
