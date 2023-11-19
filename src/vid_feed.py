import cv2
import imutils
from src import face_recognition
from scripts import send_noti

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))

        (regions, _) = hog.detectMultiScale(frame,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)

        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Extract face from the region
            face_image = frame[y:y+h, x:x+w]

            # Encode the face
            face_encoding = face_recognition.face_encodings(face_image)

            # Compare with the known face
            if len(face_encoding) > 0 and face_recognition.compare_faces([known_face_encoding], face_encoding[0])[0]:
                # Known person detected, do nothing
                pass
            else:
                # Unknown person detected, send a notification
                notification_title = 'Unknown Person Alert!'
                notification_body = 'An unknown person has been detected.'
                send_push_notification(notification_title, notification_body)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
