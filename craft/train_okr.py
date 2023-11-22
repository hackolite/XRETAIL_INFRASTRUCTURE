import zipfile
import datetime
import string
import glob
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import cv2
import keras_ocr

import os
import math
import imgaug
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf



assert tf.test.is_gpu_available()

data_dir = "."
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
alphabet = string.digits + string.ascii_letters + '!?. '
recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
fonts = keras_ocr.data_generation.get_fonts(
    alphabet=alphabet,
    cache_dir=data_dir
)
backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=data_dir)

text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet)
print('The first generated text is:', next(text_generator))

def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test

background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)



res = []
background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

c_font = []
c_background = []


for current_fonts, current_backgrounds in zip(font_splits, background_splits):
     c_font.append(current_fonts)
     c_background.append(current_backgrounds)	




image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)
    )  for current_fonts, current_backgrounds in zip(
        c_font,
        c_background
    )
]

def generate_dataset(folder=None, size=None, imagen=None) :
  ind = 0
  dataset = []
  for rs in image_generators[1]:
    ln = "./"+folder+"/"+str(ind)+".jpeg"
    cv2.imwrite(ln, rs[0])
    dataset.append((ln, rs[1], 1))
    ind += 1
    if size == ind:
        break 
  return dataset

dataset = generate_dataset(folder="images", size=1000, imagen=image_generators)


train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])

generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=train,
    augmenter=augmenter,
    **generator_kwargs
)
validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)


detector = keras_ocr.detection.Detector()

batch_size = 1
detector_batch_size = 1 
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]

detector.model.fit_generator(
    generator=training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=1000,
    workers=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5), tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5'), tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv')],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size),
    batch_size=detector_batch_size
)
