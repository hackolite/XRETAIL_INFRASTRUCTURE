import cv2
import keras_ocr
import os 
import time
import xmltodict


im_fl = os.listdir("./craft/pad")
detector = keras_ocr.detection.Detector()
detector.model.load_weights("./craft/detector_born_digital.h5")





bbox = {'name': 'text', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '215', 'ymin': '334', 'xmax': '436', 'ymax': '501'}}



def create_images(im_fl=None):
    for i in im_fl:
        if 'jpeg' in i:
            image = keras_ocr.tools.read("./craft/pad/"+i)
            xml   = i.replace(".jpeg",".xml")
            xml_path = "./craft/pad/"+xml
            try:
                with open(xml_path) as fd:
                    doc = xmltodict.parse(fd.read())
                    print(doc["annotation"]["object"])
            except Exception as e:
                print(e)  
            
            
            
            ratio_h = image.shape[0]/640
            ratio_w = image.shape[1]/640
            
            #print("ratios :", image 
            #print("ratios :", ratio_w, ratio_h) 
            
            
            image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            boxes = detector.detect(images=[image])[0]
        
            for box in boxes:
                xmin, ymin, xmax, ymax = int(box[0][0] * ratio_w) ,int(box[0][1] * ratio_h), int(box[2][0] * ratio_w), int(box[2][1] * ratio_h)
                try:
                    
                    doc["annotation"]["object"].append({'name': 'text', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}})
                    #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),7)
                except Exception as e:
                    print(e)
                    if type(doc["annotation"]["object"]) == dict:
                        doc["annotation"]["object"] = [doc["annotation"]["object"], {'name': 'text', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}}] 

                    #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),7)


            with open(xml_path, 'w') as result_file:
                result_file.write(xmltodict.unparse(doc))

            #cv2.imwrite("./craft/pad/"+i, image)

create_images(im_fl)
