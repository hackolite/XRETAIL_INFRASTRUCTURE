# -*- coding=utf-8 -*-

import os
import numpy as np
from PIL import Image
import h5py as hp


IMG_DIR="/content/XRETAIL_INFRASTRUCTURE/SynthText/dataset/bg_img"
depth = hp.File("/content/XRETAIL_INFRASTRUCTURE/SynthText/dataset/depth.h5","r")
seg_mask = hp.File("/content/XRETAIL_INFRASTRUCTURE/SynthText/dataset/seg.h5","r")
seg = seg_mask[u'mask']


dset = hp.File("/content/XRETAIL_INFRASTRUCTURE/SynthText/data/dset.h5","w")
dImage = dset.create_group("image")
dDepth = dset.create_group("depth")
dSeg = dset.create_group("seg")

if __name__ == "__main__":
    imgFiles = os.listdir(IMG_DIR)
    print(len(imgFiles))
    i = 0
    for j in [0,1]:
        for file in imgFiles[0:5000]:
            imgFile = os.path.join(IMG_DIR, file)
            if os.path.isfile(imgFile):
                if file in depth.keys() and file in seg.keys():
                    try:
                        img = Image.open(imgFile)
                        img_arr = np.array(img)
                        dImage.create_dataset(file+"_"+str(j), data=img_arr)
                        dDepth.create_dataset(file+"_"+str(j), data=depth[file][:])
                        dSeg.create_dataset(file+"_"+str(j), data=seg[file][:])
                        dSeg[file+"_"+str(j)].attrs.create("label", seg[file].attrs["label"])
                        dSeg[file+"_"+str(j)].attrs.create("area", seg[file].attrs["area"])
                        i += 1
                        if i == 10:
                            break
                    except:
                        continue


depth.close()
seg_mask.close()
dset.close()
