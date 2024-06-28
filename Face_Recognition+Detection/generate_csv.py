import os
import pandas as pd
import numpy as np
import cv2
import csv
from keras.utils import to_categorical



base_folder = "D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/LFW_DataSet"

csv_normal_file = os.path.join(base_folder, "data.csv")
csv_gray_file = os.path.join(base_folder, "data_gray.csv")

train_csv_file = os.path.join(base_folder, "peopleDevTrain.csv")
test_csv_file = os.path.join(base_folder, "peopleDevTest.csv")

train_data = pd.read_csv(train_csv_file)
test_data = pd.read_csv(test_csv_file)

train_names = train_data["name"].values
train_number_of_images = train_data["images"].values

test_names = test_data["name"].values
test_number_of_images = test_data["images"].values

names_folder = os.path.join(base_folder, "lfw-deepfunneled")


def add_header_to_csv(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["label", "name", "usage", "image"])


def save_image_to_csv(image, label, name, usage):
    global csv_normal_file, csv_gray_file
    
    
    
    pixels_string = ""
    # Folosesc flatten pentru a transforma imaginea intr-un vector unidimensional
    for pixel in image.flatten():
        pixels_string += str(pixel) + " "
    pixels_string = pixels_string[:-1]
    

    with open(csv_normal_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label, name, usage, pixels_string])
    
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    pixels_string = ""
    for pixel in image.flatten():
        pixels_string += str(pixel) + " "
    pixels_string = pixels_string[:-1]
    
    with open(csv_gray_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label, name, usage, pixels_string])


def main():
    add_header_to_csv(csv_normal_file)
    add_header_to_csv(csv_gray_file)
    
    max_index_train = 5
    
    index = 0
    
    for i in range(len(train_names)):
        name_folder = os.path.join(names_folder, train_names[i])
        
        for j in range(1, train_number_of_images[i] + 1):
            image_path = os.path.join(name_folder, train_names[i] + "_" + f"{j:04}" + ".jpg")
            
            image = cv2.imread(image_path)
            
            image = cv2.resize(image, (128, 128))
            
            save_image_to_csv(image, i, train_names[i], "train")
            # save_image_to_csv(image, i, "train")
            
            index += 1

        # if index > max_index_train:
        #     break
        
    max_index_test = 5
        
    index = 0
        
    for i in range(len(test_names)):
        name_folder = os.path.join(names_folder, test_names[i])
        
        for j in range(1, test_number_of_images[i] + 1):
            image_path = os.path.join(name_folder, test_names[i] + "_" + f"{j:04}" + ".jpg")
            
            image = cv2.imread(image_path)
            
            image = cv2.resize(image, (128, 128))
            
            save_image_to_csv(image, i, test_names[i], "test")
            # save_image_to_csv(image, i, "train")
            
            index += 1

        # if index > max_index_test:
        #     break




if __name__ == "__main__":
    main()
