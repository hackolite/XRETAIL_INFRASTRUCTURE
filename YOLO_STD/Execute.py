from keras import backend as K
import keras
import cv2
from Utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import *
from keras.applications.mobilenetv2 import MobileNetV2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os


def save_model(model):
    model_json = model.to_json()
    with open("text_detect_model.json", "w") as json_file:
        json_file.write(model_json)

        
        
def load_model(strr):        
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


def predict_func(model , inp , iou , name):
    ans = model.predict(inp)

    boxes = decode(ans[0] , 512 , 512, iou)
    print(boxes)
    img = ((inp + 1)/2)
    img = img[0]
    for i in boxes:
        i = [int(x) for x in i]
        img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , color = (0,255,0) , thickness = 2)
    plt.imshow(img)
    plt.show()
    cv2.imwrite(os.path.join('results' , str(name) + '.jpg') , img*255.0)

X = np.load('X.npy')
model_exp = load_model('text_detect_model.json')
model_exp.load_weights('text_detect.h5')
rand = np.random.randint(0,X.shape[0], size = 5)
for i in rand:
    predict_func(model_exp, X[i:i+1] , 0.1, i)



