import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

img_dir = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Img/img_align_celeba'
eval_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Eval/list_eval_partition.txt'
identity_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Anno/identity_CelebA.txt'



identities = pd.read_csv(identity_file, sep='\s+', header=None, names=['image_id', 'identity'])
identities.set_index('image_id', inplace=True)  


evals = pd.read_csv(eval_file, sep='\s+', header=None, names=['image_id', 'partition'])



# Impart datele
train_imgs = evals[evals['partition'] == 0]['image_id']
val_imgs = evals[evals['partition'] == 1]['image_id']
test_imgs = evals[evals['partition'] == 2]['image_id']


train_imgs = train_imgs[train_imgs.isin(identities.index)]
val_imgs = val_imgs[val_imgs.isin(identities.index)]
test_imgs = test_imgs[test_imgs.isin(identities.index)]


def preprocess_images(image_ids, img_dir, img_size=(64, 64)):
    images = []
    for img_id in tqdm(image_ids):
        img_path = os.path.join(img_dir, img_id)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')  # Convertire la grayscale
        img_array = img_to_array(img)
        images.append(img_array)
    images = np.array(images, dtype='float32') / 255.0
    return images

# Preprocesarea imaginilor
X_train = preprocess_images(train_imgs, img_dir)
X_val = preprocess_images(val_imgs, img_dir)
X_test = preprocess_images(test_imgs, img_dir)


np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)



# Extrag etichetele
y_train = identities.loc[train_imgs].values[:, 0]
y_val = identities.loc[val_imgs].values[:, 0]
y_test = identities.loc[test_imgs].values[:, 0]


# Creez o lista cu toate etichetele
all_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

# Fac un label encoder 
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Encodez etichetele
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)


# Salvarea etichetelor unice
all_labels = np.unique(np.concatenate([y_train, y_val, y_test]))

# Conversia etichetelor în categorii binare
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(all_labels))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(all_labels))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(all_labels))



np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

np.save('all_labels.npy', all_labels)

