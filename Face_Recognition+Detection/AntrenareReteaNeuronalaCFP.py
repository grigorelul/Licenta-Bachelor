import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def read_pair_list(pair_list_path):
    with open(pair_list_path, 'r') as file:
        pairs = file.readlines()
    return [line.strip().split() for line in pairs]

def process_pair(pair, images_base_path, pair_type):
    if len(pair) < 2:
        print(f"Invalid pair format: {pair}")
        return None, None
    img1_id = pair[0].strip()
    img2_path = pair[1].strip().replace('../Data/Images/', '').replace('/', '\\')
    
    if pair_type == 'FP':
        img1_id = int(img1_id) - 1  # Converting to zero-based index
        img1_path = os.path.join(images_base_path, f"{img1_id // 10:03d}", "profile", f"{img1_id % 10 + 1:02d}.jpg")
    elif pair_type == 'FF':
        img1_id = int(img1_id) - 1  # Converting to zero-based index
        img1_path = os.path.join(images_base_path, f"{img1_id // 10:03d}", "frontal", f"{img1_id % 10 + 1:02d}.jpg")
    
    img2_path = os.path.join(images_base_path, img2_path)

    #img1_path = img1_path.replace('\\', '/')
    #img2_path = img2_path.replace('\\', '/')
    return [img1_path, img2_path]

def load_pairs(pair_list_path, images_base_path, pair_type):
    pairs = read_pair_list(pair_list_path)
    processed_pairs = [process_pair(pair, images_base_path, pair_type) for pair in pairs]
    # Filtrare perechi invalide
    processed_pairs = [pair for pair in processed_pairs if pair[0] is not None and pair[1] is not None]
    return processed_pairs

def load_images_and_labels(pairs):
    images = []
    labels = []
    for img1_path, img2_path in pairs:
        print(f"Loading pair: {img1_path} - {img2_path}")
        try:
            img1 = img_to_array(load_img(img1_path, target_size=(64, 64), color_mode='grayscale'))
            img2 = img_to_array(load_img(img2_path, target_size=(64, 64), color_mode='grayscale'))
            images.append(np.concatenate((img1, img2), axis=2))
            labels.append(1 if img1_path.split('\\')[-3] == img2_path.split('\\')[-3] else 0)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
    return np.array(images), np.array(labels)

# Calea către dataset
base_path = 'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\cfp-dataset'
images_base_path = os.path.join(base_path, 'Data\Images')

# Căile către fișierele de perechi
pair_list_fp_path = os.path.join(base_path, 'Protocol\Pair_list_P.txt')
pair_list_ff_path = os.path.join(base_path, 'Protocol\Pair_list_F.txt')

# Încarcă perechile
fp_pairs = load_pairs(pair_list_fp_path, images_base_path, 'FP')
ff_pairs = load_pairs(pair_list_ff_path, images_base_path, 'FF')

print(f"Loaded {len(fp_pairs)} pairs from FP protocol.")
print(f"Loaded {len(ff_pairs)} pairs from FF protocol.")

# Afișează primele 5 perechi pentru verificare
print("First 5 FP pairs:")
for img1, img2 in fp_pairs[:5]:
    print(f"Image 1 path: {img1}")
    print(f"Image 2 path: {img2}")

print("First 5 FF pairs:")
for img1, img2 in ff_pairs[:5]:
    print(f"Image 1 path: {img1}")
    print(f"Image 2 path: {img2}")

# Încarcă imaginile și etichetele
images_fp, labels_fp = load_images_and_labels(fp_pairs)
images_ff, labels_ff = load_images_and_labels(ff_pairs)

"""

# Împărțire date în seturi de antrenament și testare
X_train_fp, X_test_fp, y_train_fp, y_test_fp = train_test_split(images_fp, labels_fp, test_size=0.2, random_state=42)
X_train_ff, X_test_ff, y_train_ff, y_test_ff = train_test_split(images_ff, labels_ff, test_size=0.2, random_state=42)

# Creare și antrenare model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 2)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_fp = create_model()
model_ff = create_model()

# Antrenare model pentru FP
print("Training model for FP pairs...")
model_fp.fit(X_train_fp, y_train_fp, epochs=10, batch_size=32, validation_split=0.2)

# Antrenare model pentru FF
print("Training model for FF pairs...")
model_ff.fit(X_train_ff, y_train_ff, epochs=10, batch_size=32, validation_split=0.2)

# Evaluare model
print("Evaluating model for FP pairs...")
loss_fp, accuracy_fp = model_fp.evaluate(X_test_fp, y_test_fp)
print(f"FP Model - Loss: {loss_fp}, Accuracy: {accuracy_fp}")

print("Evaluating model for FF pairs...")
loss_ff, accuracy_ff = model_ff.evaluate(X_test_ff, y_test_ff)
print(f"FF Model - Loss: {loss_ff}, Accuracy: {accuracy_ff}")
"""
