import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Dense, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Incarc datele
train_images = np.load('peopleDevTrain_images_train64x64.npy')
train_labels = np.load('peopleDevTrain_labels_train64x64.npy')
test_images = np.load('peopleDevTrain_images_test64x64.npy')
test_labels = np.load('peopleDevTrain_labels_test64x64.npy')


train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Filtrarea datelor
label_encoder = LabelEncoder()
label_encoder.fit(np.concatenate((train_labels, test_labels)))

# Encodez label-urile
train_labels_encoded = label_encoder.transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)


# Pozele sunt de marimea 224x224x3
# Definirea arhitecturii modelului CNN se regaseste si la https://ieeexplore.ieee.org/document/8721062
# model = models.Sequential([
    
#     layers.Conv2D(filters = 96, kernel_size = 11*11, strides = 4, padding = 'valid', activation='relu',),
#     layers.MaxPooling2D(),

#     layers.Conv2D(filters = 128, kernel_size = 7*7, strides = 1, padding = 'same', activation='relu'),
#     layers.MaxPooling2D(),

#     layers.Conv2D(filters = 256, kernel_size = 5*5, strides = 1, padding = 'same', activation='relu'),
#     layers.MaxPooling2D(),

#     layers.Conv2D(filters = 256, kernel_size = 3*3, strides = 1, padding = 'same', activation='relu'),
#     layers.MaxPooling2D(),

#     layers.Conv2D(filters = 384, kernel_size = 3*3, strides = 1, padding = 'same', activation='relu'),
#     layers.MaxPooling2D(),

#     layers.Dense(units = 4096, activation='relu'),
#     #layers.FullyConnected(filter = 1, units = 4096, activation='relu'),

#     layers.Dense(units = 2622,  activation='relu'),
#     layers.Softmax(),
#     layers.Dense(len(label_encoder.classes_), units = 2622, activation='relu')
#     ])

# #Definirea optimizatorului
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# # Compilarea modelului
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Sumarul modelului pentru a vedea arhitectura și numărul de parametri
# model.summary()

# # Antrenarea modelului
 # history = model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32, validation_data=(test_images, test_labels_encoded))

# # Evaluarea performanței modelului pe setul de test
# test_loss, test_acc = model.evaluate(test_images, test_labels_encoded, verbose=2)
# print(f'\nTest accuracy: {test_acc}')

# # Salvarea modelului antrenat
# model.save('face_recognition_model.h5')

#---------- Valoarea acuratetii modelului : 79.1% jtopor ----------
# model = models.Sequential([
#     layers.Conv2D(filters = 96, kernel_size = (5, 5),strides = (1,1), input_shape=(64, 64, 3)),

#     layers.MaxPooling2D(pool_size=(2, 2)),

#     Dropout(0.20),
#     layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), activation='relu'),
#     layers.MaxPooling2D(pool_size = (2, 2)),

# #     layers.Flatten(),
# #     Dense(64, activation='relu'),
# #     Softmax(),
# #     Dense(len(label_encoder.classes_), activation='softmax')

# # ])


# model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# history = model.fit(train_images, train_labels_encoded, epochs=50, batch_size=32, validation_data=(test_images, test_labels_encoded))

# test_loss, test_acc = model.evaluate(test_images, test_labels_encoded, verbose=2)

# print(f'\nTest accuracy: {test_acc}')

# model.save('face_recognition_model.h5')
