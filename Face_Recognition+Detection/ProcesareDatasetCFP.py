import os
import cv2

def load_images_and_labels(pair_file, images_path, pair_list_file, is_same=True):
    images1 = []
    images2 = []
    labels = []

    # Load the mapping of image numbers to file paths
    pair_list = {}
    with open(pair_list_file, 'r') as file:
        for idx, line in enumerate(file, start=1):
            relative_path = line.strip()
            # Remove '../Data/Images/XXX' part from the path
            parts = relative_path.split('/')
            clean_path = '/'.join(parts[3:])
            pair_list[str(idx)] = clean_path

    with open(pair_file, 'r') as file:
        for line in file:
            pair = line.strip().split(',')
            if len(pair) != 2:
                print(f"Invalid pair line: {line.strip()}")
                continue

            img1_num = pair[0].strip()
            img2_num = pair[1].strip()

            if img1_num not in pair_list or img2_num not in pair_list:
                print(f"Image number not in pair list: {img1_num}, {img2_num}")
                continue

            img1_relative_path = pair_list[img1_num]
            img2_relative_path = pair_list[img2_num]

            img1_path = os.path.abspath(os.path.join(images_path, img1_relative_path))
            img2_path = os.path.abspath(os.path.join(images_path, img2_relative_path))

            print(f"Image 1 path: {img1_path}")
            print(f"Image 2 path: {img2_path}")

            if not os.path.exists(img1_path):
                print(f"Image path does not exist: {img1_num} {img1_path}")
                continue

            if not os.path.exists(img2_path):
                print(f"Image path does not exist: {img2_num} {img2_path}")
                continue

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None:
                print(f"Failed to load image: {img1_path}")
                continue

            if img2 is None:
                print(f"Failed to load image: {img2_path}")
                continue

            images1.append(img1)
            images2.append(img2)
            labels.append(1 if is_same else 0)

    return images1, images2, labels

base_path = r"D:\Licenta-Bachelor\Face_Recognition+Detection\Datasets\cfp-dataset"
protocol_path = os.path.join(base_path, 'Protocol', 'Split')
images_path = os.path.join(base_path, 'Data', 'Images')
pair_list_f = os.path.join(base_path, 'Protocol', 'Pair_list_F.txt')
pair_list_p = os.path.join(base_path, 'Protocol', 'Pair_list_P.txt')

# Load FF protocol
ff_same_pairs = os.path.join(protocol_path, 'FF', '01', 'same.txt')
ff_diff_pairs = os.path.join(protocol_path, 'FF', '01', 'diff.txt')

ff_images1_same, ff_images2_same, ff_labels_same = load_images_and_labels(ff_same_pairs, images_path, pair_list_f, True)
ff_images1_diff, ff_images2_diff, ff_labels_diff = load_images_and_labels(ff_diff_pairs, images_path, pair_list_f, False)

# Combine same and different pairs
ff_images1 = ff_images1_same + ff_images1_diff
ff_images2 = ff_images2_same + ff_images2_diff
ff_labels = ff_labels_same + ff_labels_diff

print(f"Loaded same pairs: {len(ff_labels_same)}")
print(f"Loaded diff pairs: {len(ff_labels_diff)}")
print(f"Total pairs: {len(ff_labels)}")

# Repeat for FP protocol if needed
fp_same_pairs = os.path.join(protocol_path, 'FP', '01', 'same.txt')
fp_diff_pairs = os.path.join(protocol_path, 'FP', '01', 'diff.txt')

fp_images1_same, fp_images2_same, fp_labels_same = load_images_and_labels(fp_same_pairs, images_path, pair_list_p, True)
fp_images1_diff, fp_images2_diff, fp_labels_diff = load_images_and_labels(fp_diff_pairs, images_path, pair_list_p, False)

# Combine same and different pairs
fp_images1 = fp_images1_same + fp_images1_diff
fp_images2 = fp_images2_same + fp_images2_diff
fp_labels = fp_labels_same + fp_labels_diff

print(f"Loaded same pairs (FP): {len(fp_labels_same)}")
print(f"Loaded diff pairs (FP): {len(fp_labels_diff)}")
print(f"Total pairs (FP): {len(fp_labels)}")
