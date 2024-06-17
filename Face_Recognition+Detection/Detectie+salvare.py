import os
import numpy as np
import cv2 as cv
import face_recognition

'''import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical'''

# Citirea unei imagini si transformarea acesteia in grayscale si redimensionarea acesteia la dimensiunea 48x48 pixeli iar apoi returnarea acesteia sub forma de array si salvarea acesteia in variabila image si in folderul faces
'''
# Calea către imaginea de intrare
input_image_path = "many_faces.jpg"

# Încarcă imaginea
image = face_recognition.load_image_file(input_image_path)

# Detectează locațiile fețelor
face_locations = face_recognition.face_locations(image)

# Creează un folder pentru a salva fețele, dacă nu există deja
output_folder = "faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parcurge toate fețele detectate, afișează-le individual și le salvează în folder
for i, (top, right, bottom, left) in enumerate(face_locations):
    # Extrage regiunea feței
    face_image = image[top:bottom, left:right]
    
    # Converteste imaginea din format RGB (utilizat de face_recognition) în BGR (utilizat de cv2)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    
    # Redimensionează imaginea feței (mărind-o)
    scale_factor = 2  # Poți ajusta acest factor pentru a mări mai mult sau mai puțin
    face_image_resized = cv2.resize(face_image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Afișează imaginea feței într-o fereastră separată
    cv2.imshow(f'Face {i+1}', face_image_resized)
    
    # Salvează imaginea feței în folderul "faces"
    face_filename = os.path.join(output_folder, f"face_{i+1}.jpg")
    cv2.imwrite(face_filename, face_image_resized)

# Așteaptă apăsarea unei taste înainte de a închide ferestrele
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# Deschide camera video
vid = cv.VideoCapture(0)

# Creează un folder pentru a salva fețele, dacă nu există deja
output_folder = "faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0
while True:
    ret, frame = vid.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Converteste frame-ul din format BGR (utilizat de cv2) în RGB (utilizat de face_recognition)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detectează locațiile fețelor
    face_locations = face_recognition.face_locations(rgb_frame)

    # Parcurge toate fețele detectate, afișează-le individual și le salvează în folder
    for j, (top, right, bottom, left) in enumerate(face_locations):
        # Extrage regiunea feței
        face_image = rgb_frame[top:bottom, left:right]
        
        # Converteste imaginea din format RGB (utilizat de face_recognition) în grayscale
        face_image_gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
        
        # Redimensionează imaginea feței la 48x48 pixeli
        face_image_resized = cv.resize(face_image_gray, (48, 48))
        
        # Salvează imaginea feței în folderul "faces"
        face_filename = os.path.join(output_folder, f"face_{frame_count}_{j}.jpg")
        cv.imwrite(face_filename, face_image_resized)

    frame_count += 1

    # Afișează frame-ul curent
    cv.imshow('Video', frame)

    # Dacă se apasă tasta 'q', oprește capturarea
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează resursele
vid.release()
cv.destroyAllWindows()
