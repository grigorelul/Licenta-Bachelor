import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Calea către fișierele CSV și dataset-ul de imagini
base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/'
image_base_path = base_path + 'lfw-deepfunneled/lfw-deepfunneled/'

match_train_path = base_path + 'matchpairsDevTrain.csv'
mismatch_train_path = base_path + 'mismatchpairsDevTrain.csv'
match_test_path = base_path + 'matchpairsDevTest.csv'
mismatch_test_path = base_path + 'mismatchpairsDevTest.csv'

# Citirea fișierelor CSV
match_train = pd.read_csv(match_train_path)
mismatch_train = pd.read_csv(mismatch_train_path)
match_test = pd.read_csv(match_test_path)
mismatch_test = pd.read_csv(mismatch_test_path)

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
        name = row['name']
        image_num1 = str(row['imagenum1']).zfill(4)
        image_num2 = str(row['imagenum2']).zfill(4)
        image_path1 = os.path.join(image_base_path, name, f"{name}_{image_num1}.jpg")
        image_path2 = os.path.join(image_base_path, name, f"{name}_{image_num2}.jpg")
        image_paths1.append(image_path1)
        image_paths2.append(image_path2)
        labels.append(row['label'])
    return image_paths1, image_paths2, labels

train_image_paths1, train_image_paths2, train_labels = get_image_paths_and_labels(train_combined, image_base_path)
test_image_paths1, test_image_paths2, test_labels = get_image_paths_and_labels(test_combined, image_base_path)

# Parametrii de imagine
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Funcția pentru preprocesarea imaginilor
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalize to [0, 1]
    else:
        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))  # Placeholder for missing image
    return image

# Preprocesarea imaginilor de antrenament și testare
train_images1 = np.array([preprocess_image(path) for path in tqdm(train_image_paths1)])
train_images2 = np.array([preprocess_image(path) for path in tqdm(train_image_paths2)])
train_labels = np.array(train_labels)

test_images1 = np.array([preprocess_image(path) for path in tqdm(test_image_paths1)])
test_images2 = np.array([preprocess_image(path) for path in tqdm(test_image_paths2)])
test_labels = np.array(test_labels)

# Construirea rețelei de bază
def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)

# Funcția pentru calcularea distanței Euclidiene
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
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



'''100%
import pandas as pd
import os

# Define the paths to the CSV files
train_csv_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/matchpairsDevTrain.csv'
test_csv_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/matchpairsDevTest.csv'
image_base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled'

# Load the CSV files
train_pairs = pd.read_csv(train_csv_path)
test_pairs = pd.read_csv(test_csv_path)

# Function to get image paths from the pairs CSV
def get_image_paths_and_labels(pairs_df, image_base_path):
    image_paths1 = []
    image_paths2 = []
    labels = []
    for _, row in pairs_df.iterrows():
        name = row['name']
        image_num1 = str(row['imagenum1']).zfill(4)
        image_num2 = str(row['imagenum2']).zfill(4)
        image_path1 = os.path.join(image_base_path, name, f"{name}_{image_num1}.jpg")
        image_path2 = os.path.join(image_base_path, name, f"{name}_{image_num2}.jpg")
        image_paths1.append(image_path1)
        image_paths2.append(image_path2)
        labels.append(1 if name in image_path1 else 0)  # Assuming positive pairs, negative pairs should be handled separately
    return image_paths1, image_paths2, labels

train_image_paths1, train_image_paths2, train_labels = get_image_paths_and_labels(train_pairs, image_base_path)
test_image_paths1, test_image_paths2, test_labels = get_image_paths_and_labels(test_pairs, image_base_path)

# Print sample data
print(train_image_paths1[:5], train_image_paths2[:5], train_labels[:5])

import numpy as np
import cv2
from tqdm import tqdm

IMG_HEIGHT = 128
IMG_WIDTH = 128

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalize to [0, 1]
    else:
        image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))  # Placeholder for missing image
    return image

# Preprocess training images
train_images1 = np.array([preprocess_image(path) for path in tqdm(train_image_paths1)])
train_images2 = np.array([preprocess_image(path) for path in tqdm(train_image_paths2)])
train_labels = np.array(train_labels)

# Preprocess testing images
test_images1 = np.array([preprocess_image(path) for path in tqdm(test_image_paths1)])
test_images2 = np.array([preprocess_image(path) for path in tqdm(test_image_paths2)])
test_labels = np.array(test_labels)


'''# Construirea rețelei de bază
def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)

# Funcția pentru calcularea distanței Euclidiene
def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

input_shape = (128, 128, 3)
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
# train_images1, train_images2, train_labels sunt pregătite în mod corespunzător

history = model.fit([train_images1, train_images2], train_labels, validation_data=([test_images1, test_images2], test_labels), epochs=10, batch_size=32)

# Evaluarea modelului
test_loss, test_accuracy = model.evaluate([test_images1, test_images2], test_labels)
print(f"Acuratețea pe setul de test: {test_accuracy:.4f}")

# Plotarea istoriei antrenării
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='acuratețe')
plt.plot(history.history['val_accuracy'], label='acuratețe validare')
plt.xlabel('Epocă')
plt.ylabel('Acuratețe')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

'''Acuratete de 100% pe setul de testare'''
'''from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate

def build_siamese_model(input_shape):
    # Define the CNN model
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(input, x)
    return model

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
siamese_model = build_siamese_model(input_shape)

# Create the siamese network
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = siamese_model(input_a)
processed_b = siamese_model(input_b)

merged = concatenate([processed_a, processed_b])
output = Dense(1, activation='sigmoid')(merged)

model = Model([input_a, input_b], output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit([train_images1, train_images2], train_labels, validation_data=([test_images1, test_images2], test_labels), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([test_images1, test_images2], test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()'''