import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf

# Paths
base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/'
images_path = os.path.join(base_path, 'img_align_celeba')
attributes_path = os.path.join(base_path, 'list_attr_celeba.csv')

# Load attributes
attributes = pd.read_csv(attributes_path)
attributes.set_index('image_id', inplace=True)

# Parameters
IMG_HEIGHT = 96
IMG_WIDTH = 96

# Function to load and preprocess images
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Load all images and their attributes
image_files = os.listdir(images_path)
images = []
labels = []

for image_file in image_files:
    image_path = os.path.join(images_path, image_file)
    image = load_and_preprocess_image(image_path)
    label = attributes.loc[image_file].values  # Assuming binary attributes
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Split the dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save the datasets for later use
np.save(os.path.join(base_path, 'train_images.npy'), train_images)
np.save(os.path.join(base_path, 'test_images.npy'), test_images)
np.save(os.path.join(base_path, 'train_labels.npy'), train_labels)
np.save(os.path.join(base_path, 'test_labels.npy'), test_labels)

print('Data preparation done.')
