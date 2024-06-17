import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Calea către fișierele CSV și dataset-ul de imagini
base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/'
faces_base_path = base_path + 'lfw-deepfunneled/'

# Citirea fișierelor CSV
match_train = pd.read_csv(base_path + 'matchpairsDevTrain.csv')
mismatch_train = pd.read_csv(base_path + 'mismatchpairsDevTrain.csv')
match_test = pd.read_csv(base_path + 'matchpairsDevTest.csv')
mismatch_test = pd.read_csv(base_path + 'mismatchpairsDevTest.csv')

# Afișarea coloanelor pentru verificare
print("Coloane în matchpairsDevTrain.csv:", match_train.columns)
print("Coloane în mismatchpairsDevTrain.csv:", mismatch_train.columns)
print("Coloane în matchpairsDevTest.csv:", match_test.columns)
print("Coloane în mismatchpairsDevTest.csv:", mismatch_test.columns)

# Adăugarea etichetelor: 1 pentru perechi potrivite, 0 pentru perechi nepotrivite
match_train['label'] = 1
mismatch_train['label'] = 0
match_test['label'] = 1
mismatch_test['label'] = 0

# Combinarea perechilor
train_combined = pd.concat([match_train, mismatch_train], ignore_index=True)
test_combined = pd.concat([match_test, mismatch_test], ignore_index=True)

# Funcția pentru preluarea căilor de imagine și etichetelor
def get_image_paths_and_labels(pairs_df, image_base_path):
    image_paths1 = []
    image_paths2 = []
    labels = []
    for _, row in pairs_df.iterrows():
        if 'name.1' in row:
            name1 = str(row['name'])
            name2 = str(row['name.1'])
        else:
            name1 = str(row['name'])
            name2 = str(row['name'])
        image_num1 = str(int(row['imagenum1'])).zfill(4)
        image_num2 = str(int(row['imagenum2'])).zfill(4)
        image_path1 = os.path.join(image_base_path, name1, f"{name1}_{image_num1}.jpg")
        image_path2 = os.path.join(image_base_path, name2, f"{name2}_{image_num2}.jpg")
        if os.path.exists(image_path1) and os.path.exists(image_path2):
            image_paths1.append(image_path1)
            image_paths2.append(image_path2)
            labels.append(row['label'])
        else:
            print(f"Image not found: {image_path1} or {image_path2}")
    return image_paths1, image_paths2, labels

train_image_paths1, train_image_paths2, train_labels = get_image_paths_and_labels(train_combined, faces_base_path)
test_image_paths1, test_image_paths2, test_labels = get_image_paths_and_labels(test_combined, faces_base_path)

# Parametrii de imagine
IMG_HEIGHT = 96
IMG_WIDTH = 96

# Funcția pentru preprocesarea imaginilor
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalize to [0, 1]
    else:
        image = np.zeros((IMG_HEIGHT, IMG_WIDTH))  # Placeholder for missing image
    return image

# Preprocesarea imaginilor de antrenament și testare
train_images1 = np.array([preprocess_image(path) for path in tqdm(train_image_paths1)])
train_images2 = np.array([preprocess_image(path) for path in tqdm(train_image_paths2)])
train_labels = np.array(train_labels)

test_images1 = np.array([preprocess_image(path) for path in tqdm(test_image_paths1)])
test_images2 = np.array([preprocess_image(path) for path in tqdm(test_image_paths2)])
test_labels = np.array(test_labels)

# Verificarea datelor
print(f"Număr imagini antrenament: {len(train_images1)}")
print(f"Număr imagini testare: {len(test_images1)}")
print(f"Distribuția etichetelor în antrenament: {np.unique(train_labels, return_counts=True)}")
print(f"Distribuția etichetelor în testare: {np.unique(test_labels, return_counts=True)}")

# Construirea rețelei de bază
def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    return Model(input, x)

# Funcția pentru calcularea distanței Euclidiene
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
base_network = build_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=euclidean_distance_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# Compilarea modelului
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Antrenarea modelului
history = model.fit([train_images1, train_images2], train_labels, validation_data=([test_images1, test_images2], test_labels), epochs=10, batch_size=32)

# Evaluarea modelului
test_loss, test_accuracy = model.evaluate([test_images1, test_images2], test_labels)
print(f"Acuratețea pe setul de test: {test_accuracy:.4f}")

# Plotarea istoriei antrenării
plt.plot(history.history['accuracy'], label='acuratețe')
plt.plot(history.history['val_accuracy'], label='acuratețe validare')
plt.xlabel('Epocă')
plt.ylabel('Acuratețe')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
