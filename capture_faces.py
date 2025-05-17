import cv2
import os

# Ask for the person's name
name = input("Enter your name: ")

# Create the dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Create a subfolder inside dataset with the entered name
folder_path = os.path.join('dataset', name)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

currentFrame = 0

while True:
    ret, frame = cap.read()
    if ret:
        # Build the filename using the folder path and frame number
        filename = os.path.join(folder_path, 'frame' + str(currentFrame) + '.jpg')
        print('Creating...', filename)

        cv2.imwrite(filename, frame)
        currentFrame += 1

        #cv2.imshow('Video Frame', frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
