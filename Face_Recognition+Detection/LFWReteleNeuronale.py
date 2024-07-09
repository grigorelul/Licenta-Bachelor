# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.layers import MaxPooling1D, Conv1D, Dense, Softmax
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# import matplotlib.pyplot as plt

# # Incarc datele
# train_images = np.load('peopleDevTrain_images_train224x224.npy')
# train_labels = np.load('peopleDevTrain_labels_train224x224.npy')
# test_images = np.load('peopleDevTrain_images_test224x224.npy')
# test_labels = np.load('peopleDevTrain_labels_test224x224.npy')


# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# # Filtrarea datelor
# label_encoder = LabelEncoder()
# label_encoder.fit(np.concatenate((train_labels, test_labels)))

# # Encodez label-urile
# train_labels_encoded = label_encoder.transform(train_labels)
# test_labels_encoded = label_encoder.transform(test_labels)

# # 21.1514 acuratetea modelului dupa 20 de epoci
# # 22.83 acuratetea modelului dupa 50 de epoci
# model = models.Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(len(label_encoder.classes_), activation='softmax')
# ])

# # Compilare model
# optimizer = Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Sumar model
# model.summary()

# # Antrenarea modelului
# history = model.fit(train_images, train_labels_encoded, epochs=50, batch_size=32, validation_data=(test_images, test_labels_encoded))

# # Evaluarea performantei
# test_loss, test_acc = model.evaluate(test_images, test_labels_encoded, verbose=2)
# print(f'\nTest accuracy: {test_acc}')

# # Salvare model
# model.save('face_recognition_model0.h5')


# # Ploturi de acuratete si loss
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()


# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label = 'val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim([0, 20])
# plt.legend(loc='lower right')
# plt.show()



#Incercare cu 128x128 cu colegii de la facultate


import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Dense, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Incarc datele
train_images = np.load('1peopleDevTrain_images_train.npy')
train_labels = np.load('1peopleDevTrain_labels_train.npy')
test_images = np.load('1peopleDevTrain_images_test.npy')
test_labels = np.load('1peopleDevTrain_labels_test.npy')


train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Filtrarea datelor
label_encoder = LabelEncoder()
label_encoder.fit(np.concatenate((train_labels, test_labels)))

# Encodez label-urile
train_labels_encoded = label_encoder.transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# 21.1514 acuratetea modelului dupa 20 de epoci
# 22.83 acuratetea modelului dupa 50 de epoci
model = models.Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilare model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Sumar model
model.summary()

# Antrenarea modelului
history = model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32, validation_data=(test_images, test_labels_encoded))

# Evaluarea performantei
test_loss, test_acc = model.evaluate(test_images, test_labels_encoded, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Salvare model
model.save('1face_recognition_model.h5')


# Ploturi de acuratete si loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 20])
plt.legend(loc='lower right')
plt.show()