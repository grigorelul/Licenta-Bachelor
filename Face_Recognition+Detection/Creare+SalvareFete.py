import os
import cv2 as cv
import face_recognition

# Calea către directorul de imagini brute
raw_image_base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/lfw-deepfunneled/'

# Directorul de ieșire pentru fețele detectate
output_base_path = 'D:/Licenta-Bachelor/Face_Recognition+Detection/Datasets/lfw-dataset/faces/'

# Asigură-te că directorul de ieșire există
os.makedirs(output_base_path, exist_ok=True)

# Parcurge toate directoarele și fișierele
for person_name in os.listdir(raw_image_base_path):
    person_dir = os.path.join(raw_image_base_path, person_name)
    if os.path.isdir(person_dir):
        output_person_dir = os.path.join(output_base_path, person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_image = image[top:bottom, left:right]
                face_image = cv.cvtColor(face_image, cv.COLOR_RGB2BGR)
                face_image = cv.resize(face_image, (96, 96))
                face_output_path = os.path.join(output_person_dir, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
                cv.imwrite(face_output_path, face_image)

print("Fețele au fost detectate și salvate.")
