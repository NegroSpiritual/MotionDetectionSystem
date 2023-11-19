import cv2
import face_recognition
from pushbullet import Pushbullet

# Set up your Pushbullet API key
pushbullet_api_key = "YOUR_PUSHBULLET_API_KEY"
pb = Pushbullet(pushbullet_api_key)

# Load known faces from the dataset
known_faces = []
known_labels = []

known_faces_paths = ["dataset/person1_1.jpg", "dataset/person1_2.jpg", "dataset/person2_1.jpg"]

for path in known_faces_paths:
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_labels.append(path.split("/")[-1].split("_")[0])  # Extract label from filename

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Face recognition parameters
tolerance = 0.6  # Adjust this value based on your environment

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face in the dataset
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=tolerance)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_labels[first_match_index]

        if name == "Unknown":
            # Unauthorized person detected, send alert using Pushbullet
            push = pb.push_note("Unauthorized Movement", "Unauthorized person detected in the webcam!")

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Unauthorized Movement Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
