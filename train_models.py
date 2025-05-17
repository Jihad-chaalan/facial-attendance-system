import cv2
import os
import numpy as np
import pickle

def prepare_training_data():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Define both dataset directories to process
    data_dirs = ['dataset', 'rescaled_dataset']  # Include both folders
    
    name_to_label = {}  # Maps person names to unique labels
    current_label = 0
    faces = []
    labels = []

    for data_dir in data_dirs:
        # Skip directory if it doesn't exist
        if not os.path.exists(data_dir):
            print(f"⚠️ Directory '{data_dir}' not found. Skipping.")
            continue

        # Process each person's folder
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            
            # Skip non-directories (e.g., files)
            if not os.path.isdir(person_dir):
                continue

            # Assign a label to the person (if new)
            if person_name not in name_to_label:
                name_to_label[person_name] = current_label
                current_label += 1
            label = name_to_label[person_name]

            # Process all images for this person
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip corrupted images

                # Convert to grayscale and detect faces
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(faces_detected) == 0:
                    continue  # Skip images with no faces detected

                # Extract and resize the first detected face
                x, y, w, h = faces_detected[0]
                face_roi = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (100, 100))  # Standardize size
                
                faces.append(resized_face)
                labels.append(label)

    # Save label mappings (label -> person name)
    label_dict = {v: k for k, v in name_to_label.items()}
    with open('label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)

    return np.array(faces), np.array(labels)

# Train models (same as before)
faces, labels = prepare_training_data()

# Eigenfaces
eigen_model = cv2.face.EigenFaceRecognizer_create()
eigen_model.train(faces, labels)
eigen_model.save('eigen_model.yml')

# Fisherfaces
fisher_model = cv2.face.FisherFaceRecognizer_create()
fisher_model.train(faces, labels)
fisher_model.save('fisher_model.yml')

# LBPH
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.train(faces, labels)
lbph_model.save('lbph_model.yml')

print("✅ All models trained using BOTH 'dataset' and 'rescaled_dataset'!")