import os
import cv2 as cv
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import pyodbc
import time
from datetime import datetime
import uuid

server = 'localhost'
database = 'Licenta'
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
print("Merge conexiunea!")

model = load_model('1face_recognition_model.h5')

all_labels = np.load('1people_labels.npy')

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

threshold = 0.7  

def is_manager(user_id):
    try:
        user_id = uuid.UUID(user_id)
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return False
    cursor.execute("SELECT * FROM dbo.Managers WHERE Id=?", str(user_id))
    return cursor.fetchone() is not None

def get_attendance(user_id, is_manager):
    try:
        user_id = uuid.UUID(user_id)
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return None
    if is_manager:
        cursor.execute("SELECT * FROM dbo.Attendances WHERE ManagerId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", str(user_id))
    else:
        cursor.execute("SELECT * FROM dbo.Attendances WHERE UserId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", str(user_id))
    return cursor.fetchone()

# Verify if the attendance exists for user_id and is_manager and is unique for this day
def exists_attendance_today(user_id, is_manager):
    try:
        user_id = uuid.UUID(user_id)
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return False
    if is_manager:
        cursor.execute("SELECT * FROM dbo.Attendances WHERE ManagerId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", str(user_id))
    else:
        cursor.execute("SELECT * FROM dbo.Attendances WHERE UserId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", str(user_id))
    return cursor.fetchone() is not None # Returnează True dacă există o înregistrare pentru user_id și is_manager și este unică pentru ziua curentă

def insert_attendance(user_id, is_manager):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_id = uuid.uuid4()  # Generez un GUID nou pentru înregistrare
    try:
        user_id = uuid.UUID(user_id)  # Verific dacă ID-ul este un GUID valid
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return
    if is_manager:
        cursor.execute("INSERT INTO dbo.Attendances (Id, DataSosire, ManagerId) VALUES (?, ?, ?)", (new_id, timestamp, str(user_id)))
    else:
        cursor.execute("INSERT INTO dbo.Attendances (Id, DataSosire, UserId) VALUES (?, ?, ?)", (new_id, timestamp, str(user_id)))
    conn.commit()

def update_attendance(attendance_id):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE dbo.Attendances SET DataPlecare=? WHERE Id=?", (timestamp, attendance_id))
    conn.commit()

def get_user_id_by_name(name):
    cursor.execute("SELECT Id FROM dbo.Users WHERE Nume = ?", name)
    user = cursor.fetchone()
    if user:
        return user[0]
    cursor.execute("SELECT Id FROM dbo.Managers WHERE Nume = ?", name)
    manager = cursor.fetchone()
    if manager:
        return manager[0]
    return None

vid = cv.VideoCapture(0)

if not vid.isOpened():
    print("Eroare la deschiderea camerei")
    exit(1)

frame_count = 0
while True:
    ret, frame = vid.read()
    if not ret:
        print("Nu primesc poze")
        break
    
    if frame is None:
        print("Frame-ul e None")
        break

    face_locations = face_recognition.face_locations(frame)

    for j, (top, right, bottom, left) in enumerate(face_locations):
        face_image = frame[top:bottom, left:right]
        face_image_resized = cv.resize(face_image, (256, 256))
        
        img_array = img_to_array(face_image_resized)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        max_prediction = np.max(predictions)  

        if max_prediction < threshold:
            cv.putText(frame, 'Necunoscut', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            continue

        predicted_label = np.argmax(predictions, axis=1)  # Obțin indexul cu cea mai mare probabilitate
        predicted_identity = label_encoder.inverse_transform(predicted_label)[0]  # Obțin numele persoanei

        cv.putText(frame, predicted_identity, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        print(f"Prezicerea identitatii: {predicted_identity}")

        user_id = get_user_id_by_name(predicted_identity)
        if not user_id:
            print(f"ID-ul {predicted_identity} nu a fost găsit.")
            continue

        is_manager_flag = is_manager(user_id)
        attendance = get_attendance(user_id, is_manager_flag)
        if attendance:
            update_attendance(attendance[0])
        elif not exists_attendance_today(user_id, is_manager_flag):
            insert_attendance(user_id, is_manager_flag)

    frame_count += 1
    
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()

conn.close()
