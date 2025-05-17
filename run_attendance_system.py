import cv2
import os
import numpy as np
import pickle
import csv
from datetime import datetime

# ================== 1. Load Label Dictionary ==================
with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)

# ================== 2. Choose Model Type ==================
model_type = input("Enter model type (lbph/eigen/fisher): ").strip().lower()

if model_type == 'lbph':
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('lbph_model.yml')
    threshold = 100  # Lower confidence is better
elif model_type == 'eigen':
    model = cv2.face.EigenFaceRecognizer_create()
    model.read('eigen_model.yml')
    threshold = 4500  # Adjust based on actual test values
elif model_type == 'fisher':
    model = cv2.face.FisherFaceRecognizer_create()
    model.read('fisher_model.yml')
    threshold = 100  # Adjust based on actual test values
else:
    raise ValueError("Invalid model type. Choose from: lbph, eigen, fisher")

# ================== 3. Load Face Cascade ==================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================== 4. Helper Functions ==================
def init_video_writer(camera):
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

def init_csv_file():
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['Name', 'Timestamp'])
    return open('attendance.csv', 'a', newline='')

# ================== 5. Face Recognition Loop ==================
camera = cv2.VideoCapture(0)
video_writer = init_video_writer(camera)
csv_file = init_csv_file()
csv_writer = csv.writer(csv_file)
logged_names = set()

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        label, confidence = model.predict(face_roi)

        print(f"[DEBUG] Label: {label}, Confidence: {confidence:.2f}")

        # Decide if the prediction is acceptable
        if model_type == 'lbph':
            is_recognized = confidence < threshold
        else:  # eigen or fisher
            is_recognized = confidence < threshold

        person_name = label_dict[label] if is_recognized and label in label_dict else "Unknown"

        # Log attendance
        if is_recognized and person_name != "Unknown" and person_name not in logged_names:
            csv_writer.writerow([person_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            logged_names.add(person_name)

        # Draw rectangle and name
        color = (0, 255, 0) if is_recognized else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    video_writer.write(frame)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== 6. Cleanup ==================
camera.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()
