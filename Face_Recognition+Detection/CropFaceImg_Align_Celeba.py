import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os

img_dir = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Img/img_align_celeba'  # Calea către imagini
eval_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Eval/list_eval_partition.txt'  # Calea către fișierul de partitionare
identity_file = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/CelebA/Anno/identity_CelebA.txt'  # Calea către fișierul de identitate


# Încărcarea fișierului de identitate
identities = pd.read_csv(identity_file, sep='\s+', header=None, names=['image_id', 'identity'])
identities.set_index('image_id', inplace=True)  # Asigură-te că indexul este setat corect

# Încărcarea fișierului de evaluare
evals = pd.read_csv(eval_file, sep='\s+', header=None, names=['image_id', 'partition'])

# Verifică primele câteva valori pentru a te asigura că datele sunt corect încărcate
print("Primele câteva valori din identities:")
print(identities.head())

print("Primele câteva valori din evals:")
print(evals.head())

# Împărțirea datelor în antrenament, validare și testare
train_imgs = evals[evals['partition'] == 0]['image_id']
val_imgs = evals[evals['partition'] == 1]['image_id']
test_imgs = evals[evals['partition'] == 2]['image_id']

# Verifică dacă toate valorile din train_imgs sunt în identities
missing_train_imgs = train_imgs[~train_imgs.isin(identities.index)]
if not missing_train_imgs.empty:
    print("Imagini de antrenament lipsă din identities:")
    print(missing_train_imgs)

# Filtrează valorile pentru a te asigura că sunt prezente în indexul identităților
train_imgs = train_imgs[train_imgs.isin(identities.index)]
val_imgs = val_imgs[val_imgs.isin(identities.index)]
test_imgs = test_imgs[test_imgs.isin(identities.index)]

# Verifică lungimea seturilor de date
print(f"Număr de imagini de antrenament: {len(train_imgs)}")
print(f"Număr de imagini de validare: {len(val_imgs)}")
print(f"Număr de imagini de testare: {len(test_imgs)}")

def preprocess_images(image_ids, img_dir, img_size=(64, 64)):
    images = []
    for img_id in tqdm(image_ids):
        img_path = os.path.join(img_dir, img_id)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')  # Convertire la grayscale
        img_array = img_to_array(img)
        images.append(img_array)
    images = np.array(images, dtype='float32') / 255.0
    return images

# Preprocesarea seturilor de date
X_train = preprocess_images(train_imgs, img_dir)
X_val = preprocess_images(val_imgs, img_dir)
X_test = preprocess_images(test_imgs, img_dir)

# Verifică lungimea seturilor de imagini preprocesate
print(f"Număr de imagini de antrenament preprocesate: {len(X_train)}")
print(f"Număr de imagini de validare preprocesate: {len(X_val)}")
print(f"Număr de imagini de testare preprocesate: {len(X_test)}")

# Verifică primele câteva valori din train_imgs pentru a te asigura că indexul este corect
print("Primele câteva valori din train_imgs:")
print(train_imgs.head())

# Extrage identitățile corespunzătoare
try:
    y_train = identities.loc[train_imgs].values[:, 0]
    y_val = identities.loc[val_imgs].values[:, 0]
    y_test = identities.loc[test_imgs].values[:, 0]
except KeyError as e:
    print("A apărut un KeyError:", e)
    print("Verifică dacă toate valorile din train_imgs sunt în indexul identities")

# Verifică lungimea seturilor de etichete după extragere
print(f"Număr de etichete de antrenament extrase: {len(y_train)}")
print(f"Număr de etichete de validare extrase: {len(y_val)}")
print(f"Număr de etichete de testare extrase: {len(y_test)}")

# Creează o listă cu toate etichetele unice
all_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
print(f"Toate etichetele unice: {all_labels}")

from sklearn.preprocessing import LabelEncoder

# Inițializează și antrenează LabelEncoder pe toate etichetele
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transformă etichetele de antrenament, validare și testare
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Verifică lungimea seturilor de etichete după transformare
print(f"Număr de etichete de antrenament după transformare: {len(y_train)}")
print(f"Număr de etichete de validare după transformare: {len(y_val)}")
print(f"Număr de etichete de testare după transformare: {len(y_test)}")

# Conversia etichetelor în categorii binare
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(all_labels))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(all_labels))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(all_labels))

# Verifică lungimea seturilor de etichete după conversie
print(f"Număr de etichete de antrenament după conversie: {len(y_train)}")
print(f"Număr de etichete de validare după conversie: {len(y_val)}")
print(f"Număr de etichete de testare după conversie: {len(y_test)}")

# Crearea modelului CNN
model = Sequential()

model.add(Input(shape=(64, 64, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(all_labels), activation='softmax'))  # Numărul de clase

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

scores = model.evaluate(X_test, y_test)
print(f"Accuracy: {scores[1]*100}%")