# facial-attendance-system
# Facial Recognition Attendance System 🎯

This is a Python-based facial recognition attendance system. It uses OpenCV to detect and recognize faces, allowing for an automated way to mark attendance based on facial features.

## 🔧 Features

- Capture and store facial images of individuals
- Preprocess and organize face datasets
- Train facial recognition models (EigenFaces, FisherFaces, LBPH)
- Run real-time facial recognition for attendance
- Store attendance data with timestamps

## 📁 Project Structure

```bash
facial-attendance-system/
├── capture_faces.py           # Script to capture and save face images
├── preprocess_dataset.py      # Organize and preprocess the face dataset
├── train_models.py            # Train face recognition models
├── run_attendance_system.py   # Run the real-time attendance system
├── .gitignore                 # Git ignore file
├── requirements.txt           # Python dependencies



## ▶️ Usage
1. Capture Faces
bash
Copy
Edit
python capture_faces.py
This will open your webcam and save face images for a person into the dataset.

2. Preprocess Dataset
bash
Copy
Edit
python preprocess_dataset.py
3. Train the Model
bash
Copy
Edit
python train_models.py
4. Run Attendance System
bash
Copy
Edit
python run_attendance_system.py
This starts the real-time facial recognition system and marks attendance.
