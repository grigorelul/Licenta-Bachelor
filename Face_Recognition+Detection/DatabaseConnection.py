# '''Baza de data conexiune + inserare campuri'''
# import os
# import cv2 as cv
# import face_recognition
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import LabelEncoder
# import pyodbc
# import time
# from datetime import datetime


# server = 'localhost'
# database = 'Licenta'
# connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'


# try:
#     conn = pyodbc.connect(connection_string)
#     cursor = conn.cursor()
#     print("Conexiunea la baza de date a fost realizată cu succes!")
# except pyodbc.Error as ex:
#     print("Eroare la conectare:")
#     print(ex)
#     exit(1)


# model = load_model('face_recognition_model.h5')
# time.sleep(50)
# # Încărcarea etichetelor
# all_labels = np.load('people_labels.npy')

# label_encoder = LabelEncoder()
# label_encoder.fit(all_labels)


# threshold = 0.7  

# def is_manager(user_id):
#     cursor.execute("SELECT * FROM dbo.Managers WHERE Id=?", user_id)
#     return cursor.fetchone() is not None

# def get_attendance(user_id, is_manager):
#     if is_manager:
#         cursor.execute("SELECT * FROM dbo.Attendances WHERE ManagerId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", user_id)
#     else:
#         cursor.execute("SELECT * FROM dbo.Attendances WHERE UserId=? AND CONVERT(date, DataSosire) = CONVERT(date, GETDATE())", user_id)
#     return cursor.fetchone()

# def insert_attendance(user_id, is_manager):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     if is_manager:
#         cursor.execute("INSERT INTO dbo.Attendances (DataSosire, ManagerId) VALUES (?, ?)", (timestamp, user_id))
#     else:
#         cursor.execute("INSERT INTO dbo.Attendances (DataSosire, UserId) VALUES (?, ?)", (timestamp, user_id))
#     conn.commit()

# def update_attendance(attendance_id):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     cursor.execute("UPDATE dbo.Attendances SET DataPlecare=? WHERE Id=?", (timestamp, attendance_id))
#     conn.commit()



# vid = cv.VideoCapture(0)

# frame_count = 0
# while True:
#     ret, frame = vid.read()
#     if not ret:
#         print("Nu primesc poze")
#         break
    
   

#     # Vad unde sunt fețele în frame
#     face_locations = face_recognition.face_locations(frame)

    
#     for j, (top, right, bottom, left) in enumerate(face_locations):
#         # Luam fata din frame
#         face_image = frame[top:bottom, left:right]
        
#         #face_image_gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        
       
#         face_image_resized = cv.resize(face_image, (64, 64))
        
       
#         img_array = img_to_array(face_image_resized)
#         img_array = np.expand_dims(img_array, axis=0)  
#         img_array = img_array / 255.0  

        
#         predictions = model.predict(img_array)
#         max_prediction = np.max(predictions)  

#         if max_prediction < threshold:
#             # Fața este necunoscută
#             cv.putText(frame, 'Necunoscut', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#             cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             continue

        
#         predicted_label = np.argmax(predictions, axis=1)
#         predicted_identity = label_encoder.inverse_transform(predicted_label)[0]

#         cv.putText(frame, predicted_identity, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        
#         is_manager_flag = is_manager(predicted_identity)
#         attendance = get_attendance(predicted_identity, is_manager_flag)
#         if attendance:
#             update_attendance(attendance[0])
#         else:
#             insert_attendance(predicted_identity, is_manager_flag)

#     frame_count += 1

    
#     cv.imshow('Video', frame)
#     time.sleep(2)


# # vid.release()
# # cv.destroyAllWindows()

# conn.close()



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

try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print("Conexiunea la baza de date a fost realizată cu succes!")
except pyodbc.Error as ex:
    print("Eroare la conectare:")
    print(ex)
    exit(1)

model = load_model('face_recognition_model0.h5')

# Încarc etichetele
all_labels = np.load('people_labels.npy')

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

threshold = 0.1  

# Verifică dacă un utilizator este manager
def is_manager(user_id):
    try:
        user_id = uuid.UUID(user_id)
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return False
    cursor.execute("SELECT * FROM dbo.Managers WHERE Id=?", str(user_id))
    return cursor.fetchone() is not None


# Obțin datele de prezență pentru un utilizator
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

def insert_attendance(user_id, is_manager):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        user_id = uuid.UUID(user_id) 
    except ValueError:
        print(f"ID-ul {user_id} nu este un GUID valid.")
        return
    if is_manager:
        cursor.execute("INSERT INTO dbo.Attendances (DataSosire, ManagerId) VALUES (?, ?)", (timestamp, str(user_id)))
    else:
        cursor.execute("INSERT INTO dbo.Attendances (DataSosire, UserId) VALUES (?, ?)", (timestamp, str(user_id)))
    conn.commit()

def update_attendance(attendance_id):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE dbo.Attendances SET DataPlecare=? WHERE Id=?", (timestamp, attendance_id))
    conn.commit()

def get_user_id_by_name(name):
    # Încearcă să găsești ID-ul în tabelul dbo.Users
    cursor.execute("SELECT Id FROM dbo.Users WHERE Nume = ?", name)
    user = cursor.fetchone()
    if user:
        return user[0]

    # Dacă nu a fost găsit în dbo.Users, încearcă să găsești ID-ul în tabelul dbo.Managers
    cursor.execute("SELECT Id FROM dbo.Managers WHERE Nume = ?", name)
    manager = cursor.fetchone()
    if manager:
        return manager[0]

    # Dacă nu a fost găsit nici în dbo.Users, nici în dbo.Managers, returnează None
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
        print("Frame-ul este None")
        break


   
    # Vad unde sunt fețele în frame
    face_locations = face_recognition.face_locations(frame)

    for j, (top, right, bottom, left) in enumerate(face_locations):
        # Luam fata din frame
        face_image = frame[top:bottom, left:right]
        
        face_image_resized = cv.resize(face_image, (224, 224))
        
        img_array = img_to_array(face_image_resized)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        max_prediction = np.max(predictions)  

        if max_prediction < threshold:
            # Fața este necunoscută
            cv.putText(frame, 'Necunoscut', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            continue

        predicted_label = np.argmax(predictions, axis=1)
        predicted_identity = label_encoder.inverse_transform(predicted_label)[0]

        cv.putText(frame, predicted_identity, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        print(f"Predicted identity: {predicted_identity}")

        user_id = get_user_id_by_name(predicted_identity)
        if not user_id:
            print(f"ID-ul pentru {predicted_identity} nu a fost găsit.")
            continue

        is_manager_flag = is_manager(user_id)
        attendance = get_attendance(user_id, is_manager_flag)
        if attendance:
            update_attendance(attendance[0])
        else:
            insert_attendance(user_id, is_manager_flag)

    frame_count += 1
    
    cv.imshow('Video', frame)
    # cv.imwrite(f'processed_frame_{frame_count}.jpg', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()

conn.close()