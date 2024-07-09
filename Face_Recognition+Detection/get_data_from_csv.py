import os
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.optimizers import Adam
from keras.layers import Dense, Softmax, Dropout, Flatten, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_folder = "D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/LFW_DataSet"

# ---------- Normal Images ----------

csv_normal_file = os.path.join(base_folder, "data.csv")

normal_data = pd.read_csv(csv_normal_file)
normal_train_data = normal_data[normal_data["usage"] == "train"]
#normal_test_data = normal_data[normal_data["usage"] == "test"]

normal_raw_train_images = normal_train_data["image"].values
#normal_raw_test_images = normal_test_data["image"].values

normal_raw_train_labels = normal_train_data["label"].values
#normal_raw_test_labels = normal_test_data["label"].values

normal_raw_names = normal_train_data["name"].values

def process_normal_images(images):
    new_images = []
    for image in images:
        pixels = image.split(" ")
        pixels = np.array(pixels, dtype=np.uint8)
        pixels = pixels.reshape((128, 128, 3))
        new_images.append(pixels)
    return np.array(new_images)

normal_train_images = process_normal_images(normal_raw_train_images)
#normal_test_images = process_normal_images(normal_raw_test_images)

normal_train_labels = to_categorical(normal_raw_train_labels, len(set(normal_raw_train_labels)))
#normal_test_labels = to_categorical(normal_raw_test_labels, len(set(normal_raw_test_labels)))

# ---------- Gray Images ----------

csv_gray_file = os.path.join(base_folder, "data_gray.csv")

gray_data = pd.read_csv(csv_gray_file)
gray_train_data = gray_data[gray_data["usage"] == "train"]
# gray_test_data = gray_data[gray_data["usage"] == "test"]

gray_raw_train_images = gray_train_data["image"].values
# gray_raw_test_images = gray_test_data["image"].values

gray_raw_train_labels = gray_train_data["label"].values
#gray_raw_test_labels = gray_test_data["label"].values

def process_gray_images(images):
    new_images = []
    for image in images:
        pixels = image.split(" ")
        pixels = np.array(pixels, dtype=np.uint8)
        pixels = pixels.reshape((128, 128, 1))
        new_images.append(pixels)
    return np.array(new_images)

gray_train_images = process_gray_images(gray_raw_train_images)
#gray_test_images = process_gray_images(gray_raw_test_images)

gray_train_labels = to_categorical(gray_raw_train_labels, len(set(gray_raw_train_labels)))
#gray_test_labels = to_categorical(gray_raw_test_labels, len(set(gray_raw_test_labels)))



label_encoder = LabelEncoder()
label_encoder.fit(normal_raw_names)
train_labels = label_encoder.transform(normal_raw_names)
num_classes = len(set(train_labels))



#---------- CNN ----------
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



train_set_size = int(0.8 * len(normal_train_images))
train_images = normal_train_images[:train_set_size]
train_labels = normal_train_labels[:train_set_size]
test_images = normal_train_images[train_set_size:]
test_labels = normal_train_labels[train_set_size:]


train_images = train_images / 255.0
test_images = test_images / 255.0



# train_set_size = int(0.8 * len(gray_train_images))
# train_images = gray_train_images[:train_set_size]
# train_labels = gray_train_labels[:train_set_size]
# test_images = gray_train_images[train_set_size:]
# test_labels = gray_train_labels[train_set_size:]



history = model.fit(train_images, train_labels, batch_size=32,
                    epochs=10,
                    validation_data=(test_images, test_labels))


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
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()



model.save('face_recognition_model.h5')



# ---------- Main ----------

def main():
    for i in range(len(normal_train_images)):
        print(normal_train_images[i].shape, normal_train_labels[i])
        
        cv2.imshow("Normal", normal_train_images[i])
        cv2.waitKey(0)
        
    for i in range(len(gray_train_images)):
        print(gray_train_images[i].shape, gray_train_labels[i])
        
        cv2.imshow("Gray", gray_train_images[i])
        cv2.waitKey(0)
        

if __name__ == "__main__":
    main()
