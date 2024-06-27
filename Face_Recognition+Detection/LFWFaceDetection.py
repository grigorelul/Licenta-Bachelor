
"Prelucrez pozele astfel incat sa am doar fata in imagine"
"https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html - tutorial pentru detectarea fetei folosind Yunet"
"https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet - modelul Yunet pentru detectarea fetei"

import os
import cv2 as cv
import numpy as np
from yunet import YuNet


input_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\LFW_DataSet\lfw-deepfunneled'
modelPath = 'face_detection_yunet_2023mar.onnx'

yunet = YuNet(modelPath=modelPath,
              inputSize=[250, 250],
              confThreshold=0.7,
              nmsThreshold=0.3,
              topK=5000,
              backendId=0,
              targetId=0)

# Funcția de procesare a unei imagini
def process_image(image_path):
    print(f'Processing image: {image_path}')
    
    # Pun imaginea in image
    image = cv.imread(image_path)

    if image.shape[0] != 250 or image.shape[1] != 250:
        return
    
    # Folosesc functia infer pentru a detecta fetele din imagine
    detections = yunet.infer(image)
    
    if detections.shape[0] > 0:
        
        detection = detections[0]
        x, y, lungime, inaltime = (detection[0], detection[1], detection[2], detection[3])
        x, y = abs(int(x)), abs(int(y))
        lungime, inaltime = int(lungime), int(inaltime)
        
        # Luam doar fata din imagine
        face_image = image[y:y+inaltime, x:x+lungime]
       
        # Salvam peste imaginea originala
        cv.imwrite(image_path, face_image)

# Funcția care parcurge toate pozele din folder
def process_all_images(input_dir):

    # Parcurgem toate folderele din input_dir
    folders = os.listdir(input_dir)
    for folder in folders:
        folder_path = os.path.join(input_dir, folder)
        files = os.listdir(folder_path)
        for file in files:
            if not file.endswith('.jpg'):
                continue
            image_path = os.path.join(folder_path, file)
            process_image(image_path)


def resize_all_images(input_dir):
    folders = os.listdir(input_dir)
    for folder in folders:
        folder_path = os.path.join(input_dir, folder)
        files = os.listdir(folder_path)
        for file in files:
            if not file.endswith('.jpg'):
                continue
            image_path = os.path.join(folder_path, file)
            image = cv.imread(image_path)
            resized_image = cv.resize(image, (128, 128))
            cv.imwrite(image_path, resized_image)



# Procesez toate fotografiile din folder
process_all_images(input_dir)

# Redimensionez toate pozele din folder
resize_all_images(input_dir)