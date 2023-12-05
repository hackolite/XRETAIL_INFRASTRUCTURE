import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time
detector = keras_ocr.detection.Detector()
detector.model.load_weights("ssd_2.h5")

PATHTEST = "./TEST"
RESULT  = "./RES_2"


for i in os.listdir(PATHTEST):
    if "jpeg" or "jpg" in i:
        start = time.time()
        image = keras_ocr.tools.read(PATHTEST+"/"+i)
        h, w, _ = image.shape
        w = w * 2
        if h > 640:
            h=640
        if w > 640:
            w=320 
        image = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        boxes = detector.detect(images=[image])[0]
        drawn = keras_ocr.tools.drawBoxes(image=image, boxes=boxes)
        cv2.imwrite(RESULT+"/"+i, drawn)
        print(time.time() - start) 
