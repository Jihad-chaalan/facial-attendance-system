import cv2
import os

# Input and output folders
input_root = 'dataset'
output_root = 'rescaled_dataset'

# List of scales (you can change/add more)
scales = [(100, 100), (150, 150), (200, 200)]

# Create output root folder if it doesn't exist
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Loop through each person in the dataset
for person_name in os.listdir(input_root):
    input_folder = os.path.join(input_root, person_name)
    output_person_folder = os.path.join(output_root, person_name)

    if not os.path.exists(output_person_folder):
        os.makedirs(output_person_folder)

    # Loop through each image for that person
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # For each scale, resize and save
        for width, height in scales:
            resized_img = cv2.resize(img, (width, height))
            # Name example: frame0_100x100.jpg
            new_name = f"{os.path.splitext(img_name)[0]}_{width}x{height}.jpg"
            save_path = os.path.join(output_person_folder, new_name)
            cv2.imwrite(save_path, resized_img)

print("✅ Rescaling complete. Images saved in 'rescaled_dataset/'")
