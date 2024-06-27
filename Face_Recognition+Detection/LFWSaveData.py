"Verificarea și salvarea seturilor de date pentru antrenare și testare a rețelei neuronale. in format .npy"

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

data_dir = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/LFW_DataSet/lfw-deepfunneled'

peopleDevTrain_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/LFW_DataSet/peopleDevTrain.csv'
peopleDevTrain = np.genfromtxt(peopleDevTrain_path, delimiter=',', dtype=str, skip_header=1)

people_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/LFW_DataSet/people.csv'
people = np.genfromtxt(people_path, delimiter=',', dtype=str, skip_header=1)

# Funcția de încărcare a fotografiilor și etichetelor
def load_images_and_labels(dataset):
    images = []
    labels = []
    
    for person in dataset:
        if len(person) != 2:
            continue  
        
        name = person[0].strip()  
        num_images_str = person[1].strip()  
        
        if not num_images_str.isdigit():
            continue
        
        num_images = int(num_images_str)
        person_folder = os.path.join(data_dir, name)
        
        for i in range(1, num_images + 1):
            image_path = os.path.join(person_folder, f'{name}_{i:04}.jpg') # Path-ul imaginii
            print(f'Loading image: {image_path}')
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(name)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


train_images, train_labels = load_images_and_labels(peopleDevTrain)
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


#Salvam toate seturile

np.save('peopleDevTrain_images_train.npy', train_images)
np.save('peopleDevTrain_labels_train.npy', train_labels)
np.save('peopleDevTrain_images_test.npy', test_images)
np.save('peopleDevTrain_labels_test.npy', test_labels)


people_images, people_labels = load_images_and_labels(people)
np.save('people_images.npy', people_images)
np.save('people_labels.npy', people_labels)

print('Salvarea seturilor de date a fost finalizată.')
