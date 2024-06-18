import os
import cv2

'''Le fac resize la 64x64 pentru a fi compatibile cu celelalte imagini din celelalte dataseturi'''
def resize_and_save_images(source_folder, target_folder, size=(64, 64)):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                if img is not None:
                    resized_img = cv2.resize(img, size)
                    relative_path = os.path.relpath(file_path, source_folder)
                    save_path = os.path.join(target_folder, relative_path)
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv2.imwrite(save_path, resized_img)
                    print(f"Saved resized image to: {save_path}")
                else:
                    print(f"Failed to load image: {file_path}")

source_folder = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\cfp-dataset\Data\Images'
target_folder = r'D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\cfp-dataset\Resized_Images'

resize_and_save_images(source_folder, target_folder, size=(64, 64))
