import os
import cv2 as cv
import numpy as np
import pandas as pd


base_folder = "D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets"

''' Load the images from the folder '''
lfw_folder_images = os.path.join(base_folder, "lfw-dataset\lfw-deepfunneled")

lfw_dataset_folder = os.path.join(base_folder, "\lfw-dataset\")




def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_images_from_folder_with_labels(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            labels.append(filename.split('_')[0])
    return images, labels

def load_images_from_folder_with_labels_and_names(folder):
    images = []
    labels = []
    names = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            labels.append(filename.split('_')[0])
            names.append(filename.split('_')[1])
    return images, labels, names

def main ():


    if __name__ == "__main__":
        main()