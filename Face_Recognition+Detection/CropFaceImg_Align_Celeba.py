import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os
import tempfile


# Setează variabilele de mediu pentru TEMP și TMP
tempdir = 'D:/TEMP'
os.environ['TEMP'] = tempdir
os.environ['TMP'] = tempdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce nivelul de logare
os.environ['TF_TEMP'] = tempdir
os.environ['TF_CACHE_DIR'] = tempdir

# Setează directoarele temporare pentru modulul tempfile
tempfile.tempdir = tempdir

import tensorflow as tf

# Verifică dacă există GPU și configurează memoria corespunzător
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU found and memory growth set: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU")

# Verifică dacă directoarele temporare sunt setate corect
print(f"Python TEMP directory: {tempfile.gettempdir()}")
print(f"TensorFlow TEMP directory: {os.getenv('TF_TEMP')}")
print(f"TensorFlow CACHE directory: {os.getenv('TF_CACHE_DIR')}")






img_dir = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Img/img_align_celeba'  # Calea către imagini
eval_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Eval/list_eval_partition.txt'  # Calea către fișierul de partitionare
identity_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Anno/identity_CelebA.txt'  # Calea către fișierul de identitate



# Încărcarea fișierului de identitate
identities = pd.read_csv(identity_file, delim_whitespace=True, header=None, names=['image_id', 'identity'])

# Încărcarea fișierului de partitionare
evals = pd.read_csv(eval_file, delim_whitespace=True, header=None, names=['image_id', 'partition'])

# Împărțirea datelor în antrenament, validare și testare
train_imgs = evals[evals['partition'] == 0]['image_id']
val_imgs = evals[evals['partition'] == 1]['image_id']
test_imgs = evals[evals['partition'] == 2]['image_id']



def preprocess_images(image_ids, img_dir, img_size=(178, 218)):
    images = []
    for img_id in tqdm(image_ids):
        img_path = os.path.join(img_dir, img_id)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)
    images = np.array(images, dtype='float32') / 255.0
    return images

# Preprocesarea seturilor de date
X_train = preprocess_images(train_imgs, img_dir)
X_val = preprocess_images(val_imgs, img_dir)
X_test = preprocess_images(test_imgs, img_dir)

y_train = identities.loc[train_imgs].values[:, 1]
y_val = identities.loc[val_imgs].values[:, 1]
y_test = identities.loc[test_imgs].values[:, 1]


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Conversia etichetelor în categorii binare
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)


# Crearea modelului CNN

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(178, 218, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Numărul de clase

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)


scores = model.evaluate(X_test, y_test)
print(f"Accuracy: {scores[1]*100}%")