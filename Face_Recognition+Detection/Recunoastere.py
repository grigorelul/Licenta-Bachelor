import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from deepface import DeepFace

# Define paths to dataset and split files
data_dir = "LFW_DataSet/lfw_funneled"
train_pairs_file = "LFW_DataSet/pairsDevTrain.txt"
test_pairs_file = "LFW_DataSet/pairsDevTest.txt"

# Creează liste pentru a stoca imaginile și etichetele
images = []
labels = []

# Parcurge toate folderele din data_dir
for person_name in os.listdir(data_dir):
    person_folder = os.path.join(data_dir, person_name)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (48, 48))
            images.append(image)
            labels.append(person_name)

# Converteste listele în array-uri numpy
images = np.array(images)
labels = np.array(labels)

# Converteste etichetele în format numeric
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Împarte datele în seturi de antrenament și testare la nivel de persoane
unique_labels = np.unique(numeric_labels)
train_labels, test_labels = train_test_split(unique_labels, test_size=0.2, random_state=42)

train_images = []
train_labels_final = []
test_images = []
test_labels_final = []

# Asociază fiecare imagine cu setul corespunzător (antrenament sau testare)
for image, label in zip(images, numeric_labels):
    if label in train_labels:
        train_images.append(image)
        train_labels_final.append(label)
    else:
        test_images.append(image)
        test_labels_final.append(label)

# Converteste listele în array-uri numpy
train_images = np.array(train_images)
train_labels_final = np.array(train_labels_final)
test_images = np.array(test_images)
test_labels_final = np.array(test_labels_final)

# Verifică dacă toate etichetele sunt prezente în ambele seturi
train_classes = np.unique(train_labels_final)
test_classes = np.unique(test_labels_final)

missing_classes = set(train_classes) - set(test_classes)
if missing_classes:
    print(f"Lipsesc clasele din setul de testare: {missing_classes}")
    # Adăugăm clasele lipsă în setul de testare
    for cls in missing_classes:
        test_images = np.append(test_images, [train_images[train_labels_final == cls][0]], axis=0)
        test_labels_final = np.append(test_labels_final, [cls], axis=0)

# Converteste etichetele în format one-hot
train_labels_final = to_categorical(train_labels_final, num_classes=len(label_encoder.classes_))
test_labels_final = to_categorical(test_labels_final, num_classes=len(label_encoder.classes_))

# Verifică formele datelor încărcate
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels_final.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels_final.shape)

# Definește modelul CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Afiseaza sumarul modelului
model.summary()

# Compilează modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenează modelul
history = model.fit(train_images, train_labels_final, epochs=40, batch_size=64, validation_data=(test_images, test_labels_final))

# Evaluează modelul pe setul de testare
test_loss, test_accuracy = model.evaluate(test_images, test_labels_final, verbose=2)
print(f'Test accuracy: {test_accuracy}')

# Salvează modelul
model.save('face_recognition_model.h5')

'''
# Compilează modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenează modelul
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluează modelul pe setul de testare
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
'''
