import os
from PIL import Image
import numpy as np
from mtcnn import MTCNN

# Specifică directorul cu pozele și directorul unde să salvezi imaginile procesate
input_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\img_align_celeba'
output_dir = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\CelebA\Img\output_faces'
os.makedirs(output_dir, exist_ok=True)

# Numele fișierului de la care să înceapă procesarea
start_file = '028987.jpg'
start_processing = False

# Initializează detectorul MTCNN
detector = MTCNN()

# Funcție pentru a procesa imaginile
def process_images(input_dir, output_dir, start_file):
    global start_processing
    # Obține toate fișierele din directorul de intrare
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    for image_file in image_files:
        # Verificăm dacă am ajuns la fișierul de start
        if image_file == start_file:
            start_processing = True
        
        if not start_processing:
            continue

        image_path = os.path.join(input_dir, image_file)
        
        # Încarcă imaginea folosind PIL
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Detectează fețele în imagine
        detections = detector.detect_faces(image_array)
        
        # Dacă este detectată o față
        if detections:
            # Presupunem că există o singură față în fiecare imagine
            detection = detections[0]
            x, y, width, height = detection['box']
            x, y = abs(x), abs(y)
            face_image = image_array[y:y+height, x:x+width]
            
            # Convertește imaginea la formatul PIL pentru redimensionare și grayscale
            pil_image = Image.fromarray(face_image)
            
            # Redimensionează imaginea (exemplu: 128x128)
            resized_image = pil_image.resize((128, 128))
            
            # Convertim imaginea în grayscale
            grayscale_image = resized_image.convert('L')
            
            # Salvează imaginea procesată
            output_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}.jpg')
            grayscale_image.save(output_path)
            print(f'Saved grayscale face image: {output_path}')
        else:
            print(f'No face detected in image: {image_path}')

# Apelează funcția pentru a procesa imaginile
process_images(input_dir, output_dir, start_file)