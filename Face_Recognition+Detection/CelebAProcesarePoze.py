"Procesarea pozelor cu YuNet"

# import os
# from PIL import Image
# import numpy as np
# import cv2
# from yunet import YuNet

# input_dir = r'C:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\img_align_celeba'
# output_dir = r'C:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\output_faces'
# os.makedirs(output_dir, exist_ok=True)

# modelPath = 'face_detection_yunet_2023mar.onnx'
# yunet = YuNet(modelPath=modelPath,
#               inputSize=[178, 218],  
#               confThreshold=0.6,  # 0,7, 0.6, 0.5, 0.4 sunt valorile care au mers pentru a mai identifica fete
#               nmsThreshold=0.3,
#               topK=5000,
#               backendId=0,
#               targetId=0)

# # Funcția de ajustare a contrastului si a luminozitatii, iar valorile separarte sunt cele care au mers pentru a mai identifica fete
# # def adjust_image(image):
# #     alpha = 2.5 # Contrast control (1.0-3.0)     1.5 si 5,     0.5 si 5,      0.6 cu 2.5 si 5
# #     beta =  5   # Brightness control (0-100)
# #     adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# #     return adjusted


# def process_image(image_path, output_dir):
   
#     image = cv2.imread(image_path)
    
    
#     #adjusted_image = adjust_image(image)
    
    
#     #yunet.setInputSize([adjusted_image.shape[1], adjusted_image.shape[0]])


#     detections = yunet.infer(image)
    
#     if detections.shape[0] > 0:
#         for detection in detections:
#             x, y, width, height = (detection[0], detection[1], 
#                                    detection[2], detection[3])
#             x, y = abs(int(x)), abs(int(y))
#             face_image = image[y:y+int(height), x:x+int(width)]
            

#             pil_image = Image.fromarray(face_image)
            

#             resized_image = pil_image.resize((64, 64))
            
            
#             grayscale_image = resized_image.convert('L')
            
            
#             output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}.jpg')
#             grayscale_image.save(output_path)
#             print(f'Gray: {output_path}')
#     else:
#         print(f'Nici-o fata: {image_path}')



# def process_missing_images(input_dir, output_dir):
    
#     #Iau toate pozele din directorul de intrare
#     input_files = set([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
#     #Iau toate fotografiile din directorul de iesire
#     output_files = set([os.path.splitext(f)[0] + '.jpg' for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
#     missing_files = input_files - output_files
#     print(f'Poza lipsa {len(missing_files)}')
    
#     # Procesez imaginile lipsa
#     for missing_file in missing_files:
#         image_path = os.path.join(input_dir, missing_file)
#         process_image(image_path, output_dir)


# process_missing_images(input_dir, output_dir)


"Procesarea pozelor cu MTCNN"
# import os
# from PIL import Image
# import numpy as np
# from mtcnn import MTCNN


# input_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\img_align_celeba'
# output_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\output_faces'
# os.makedirs(output_dir, exist_ok=True)


# detector = MTCNN()

# def process_images(input_dir, output_dir):

#     # Obțin toate fișierele din directorul de intrare
#     image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
#     # Parcurg toate pozele
#     for image_file in image_files:
#         image_path = os.path.join(input_dir, image_file)
        
        
#         image = Image.open(image_path)
#         image_array = np.array(image)
        
#         detections = detector.detect_faces(image_array)
        
        
#         if detections:            # Stiu ca in toate pozele am deja doar o fata deci pot să iau doar prima detectie
#             detection = detections[0]
#             x, y, width, height = detection['box']
#             x, y = abs(x), abs(y)
#             face_image = image_array[y:y+height, x:x+width]
            
#             pil_image = Image.fromarray(face_image)
            
#             resized_image = pil_image.resize((128, 128))
            
#             grayscale_image = resized_image.convert('L')
            
#             output_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}.jpg')
#             grayscale_image.save(output_path)
#             print(f'Imagine salvata: {output_path}')
#         else:
#             print(f'Nu avem fete: {image_path}')

# process_images(input_dir, output_dir)


# " In caz ca se blocheaza la o anumita poza putem sa oprim procesarea si sa continuam de la poza respectiva"
# import os
# from PIL import Image
# import numpy as np
# from mtcnn import MTCNN

# input_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\img_align_celeba'
# output_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\output_faces'
# os.makedirs(output_dir, exist_ok=True)

# start_file = '028987.jpg'
# start_processing = False

# detector = MTCNN()

# # Funcția de procesare a imaginilor
# def process_images(input_dir, output_dir, start_file):
#     global start_processing

#     image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
#     for image_file in image_files:
        
#         if image_file == start_file:  # Verific dacă am ajuns la fișierul de start iar daca da setam start_processing pe True pentru a incepe procesarea imaginilor
#             start_processing = True
        
#         if not start_processing:
#             continue

#         image_path = os.path.join(input_dir, image_file)
        
#         image = Image.open(image_path)
#         image_array = np.array(image)
        
#         detections = detector.detect_faces(image_array)
        
#         if detections:
#             detection = detections[0]
#             x, y, width, height = detection['box']
#             x, y = abs(x), abs(y)
#             face_image = image_array[y:y+height, x:x+width]
            
#             pil_image = Image.fromarray(face_image)
            
#             resized_image = pil_image.resize((128, 128))
            
#             grayscale_image = resized_image.convert('L')
            
#             output_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}.jpg')
#             grayscale_image.save(output_path)
#             print(f'Imagine salvata: {output_path}')
#         else:
#             print(f'Nu exista fata: {image_path}')

# process_images(input_dir, output_dir, start_file)