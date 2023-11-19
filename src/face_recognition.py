import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import mediapipe as mp

# Function to load images and labels from the dataset
def load_dataset(dataset_folder):
    images = []
    labels = []

    for label in os.listdir(dataset_folder):
        label_folder = os.path.join(dataset_folder, label)

        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    images.append(img.flatten())
                    labels.append(label)

    return np.array(images), np.array(labels)

# Create a folder for the dataset
dataset_folder = 'dataset'

# Load images and labels from the dataset
images, labels = load_dataset(dataset_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Initialize the video capture from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Initialize face detection with MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

while True:
    # Capture frame from the video feed
    ret, frame = cap.read()

    if ret is False:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use face detection to get bounding boxes of faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for face in results.detections:
            face_react = np.multiply(
                [
                    face.location_data.relative_bounding_box.xmin,
                    face.location_data.relative_bounding_box.ymin,
                    face.location_data.relative_bounding_box.width,
                    face.location_data.relative_bounding_box.height,
                ],
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)

            # Extract the face region from the frame
            face_img = frame[face_react[1]:face_react[1] + face_react[3], face_react[0]:face_react[0] + face_react[2]]

            # Convert the face region to grayscale for face recognition
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Flatten the face to match the format of the training data
            flattened_face = gray_face.flatten()

            # Use the trained classifier to predict the label of the face
            predicted_label = knn_classifier.predict([flattened_face])

            # Draw the predicted label on the frame
            cv2.putText(frame, f"Predicted: {predicted_label[0]}", (face_react[0], face_react[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw bounding box around the face
            cv2.rectangle(frame, (face_react[0], face_react[1]), (face_react[0] + face_react[2], face_react[1] + face_react[3]),
                          color=(255, 255, 255), thickness=2)

    # Display the frame
    cv2.imshow('Facial Recognition', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
