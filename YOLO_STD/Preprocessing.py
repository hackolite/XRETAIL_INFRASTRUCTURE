import numpy as np
import pandas as pd
import cv2
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt



image_path = '/content/render_img'
label_path = '/content/render_lbl'


X_final = []
Y_final = []
grid_h = 16
grid_w = 16
img_w = 512
img_h = 512

global x_sl
global y_sl


import sys


def create_img(img_path=None):
    X = []
    global x_sl
    global y_sl
    
    images_files = os.listdir(img_path)
    for img_fl in images_files:
        try:
          img_fl = image_path +"/"+ img_fl
          x = cv2.imread(img_fl)
          x_sl = 512/x.shape[1]
          y_sl = 512/x.shape[0]
          img = cv2.resize(x,(img_w,img_h))
          X.append(img)
        except Exception as e:
          print(img_fl)
          print(e)
    return X

import traceback

def create_fl(label_path=None):
    Y_F = []
    labels_files = os.listdir(label_path)
    for lbl_fl in labels_files:
        try:
          lbl_fl = label_path +"/"+ lbl_fl
          name = open(lbl_fl , 'r')
          data = name.read()
          data = data.split("\n")
          data = data[:-1]
          Y = np.zeros((grid_h,grid_w,1,5))
    
          for strng in data:
            bounding_box = [int(f) for f in strng.split(",")[0:8]]
            xmin = bounding_box[0] * x_sl
            xmax = bounding_box[2] * x_sl
            ymin = bounding_box[1] * y_sl
            ymax = bounding_box[3] * y_sl

            w = (xmax - xmin)/img_w
            h = (ymax - ymin)/img_h
            
            x = ((xmax + xmin)/2)/img_w
            y = ((ymax + ymin)/2)/img_h
            x = x * grid_w
            y = y * grid_h
            
            if x >= 16:
              x = 15

            if y >= 16:
               y = 15   

            Y[int(y),int(x),0,0] = 1
            Y[int(y),int(x),0,1] = x - int(x)
            Y[int(y),int(x),0,2] = y - int(y)
            Y[int(y),int(x),0,3] = w
            Y[int(y),int(x),0,4] = h

          Y_F.append(Y)  
        
        except Exception as e:
          print(e, x, y)
    return Y_F


img_dataset = create_img(image_path)
lbl_dataset = create_fl(label_path)
X = np.array(img_dataset)
Y = np.array(lbl_dataset)
X = (X - 127.5)/127.5

np.save('./X.npy',X)
np.save('./Y.npy',Y)


