import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pandas as pd
from PIL import Image
from PIL.ImageDraw import Draw


import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16, resnet50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model



data = []
target = []
file_names = []



ann_path = "./CALTECH101_STOP_SIGN/stop_sign_annotations_converted.txt"
rows = open(ann_path).read().strip().split("\n")



for idx, row in enumerate(rows):

    # break the row to the file_name
    # and coordinates of bounding box
    row = row.split(",")

    file_name = row[0]
    file_name = file_name.split(".")[0]
    file_name = file_name.split("_")[1]
    file_name = "image_" + file_name + ".jpg"

    coords = row[1]
    coords = coords.split(" ")

    # we have unusual last line,
    # so there will be an if-else

    if (idx != 63):
        coords = coords[1:-1]
    else:
        coords = coords[1:]

    # convert to int
    coords = [int(c) for c in coords]
    print(coords)
    # read image
    path = "./CALTECH101_STOP_SIGN/stop_sign/"
    full_path = path + file_name
    img = cv2.imread(full_path)
    (h, w) = img.shape[:2]

    # scale the bounding box coordinates
    # relative to the dimensions of the img
    Xmin = float(coords[0]) / w
    Ymin = float(coords[1]) / h
    Xmax = float(coords[2]) / w
    Ymax = float(coords[3]) / h

    # load the image again with
    # tensorflow and preprocess it

    #print(Xmin, Ymin, Xmax, Ymax)

    img = load_img(full_path, target_size=(224, 224))
    img = img_to_array(img)

    data.append(img)
    target.append((Xmin, Ymin, Xmax, Ymax))
    file_names.append(file_name)


# normalize data, scaling from [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
target = np.array(target, dtype="float32")



split = train_test_split(data, target, file_names, test_size=0.10, random_state=45)

(train_imgs, test_imgs) = split[:2]
(train_target, test_target) = split[2:4]
(train_filenames, test_filenames) = split[4:]

f = open("test_images.txt", "w")
f.write("\n".join(test_filenames))
f.close()



vgg_model = VGG16(weights="imagenet",
                  include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))


vgg_model.trainable = False


flatten = vgg_model.output
flatten = Flatten()(flatten)


bbox_head = Dense(128, activation="relu")(flatten)
bbox_head = Dense(64, activation="relu")(bbox_head)
bbox_head = Dense(32, activation="relu")(bbox_head)
bbox_head = Dense(4, activation="sigmoid")(bbox_head)


model = Model(inputs=vgg_model.input, outputs=bbox_head)

LR = 1e-4
EPOCHS = 300
BATCH_SIZE = 32


opt = Adam(lr=LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())


H = model.fit(train_imgs,
              train_target,
              validation_data=(test_imgs, test_target),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1)


model.save("model_stop_signs_caltech101.h5", save_format="h5")

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, EPOCHS),
         H.history["loss"],
         label="train_loss")

plt.plot(np.arange(0, EPOCHS),
         H.history["val_loss"],
         label="val_loss")

plt.title("Bounding Box Regression Loss")
plt.xlabel("Epoch №")
plt.ylabel("Loss")
plt.legend()
plt.savefig("stop_sign_model_loss_plot.png")























































































