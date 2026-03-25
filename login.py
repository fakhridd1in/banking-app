import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def get_face_embedding(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
    return face.flatten().reshape(1, -1), (x, y, w, h)

def load_registered_face():
    folder = "my_face"
    embeddings = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(f"{folder}/{filename}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                embeddings.append(face.flatten())
    return np.array(embeddings)

print("Loading your face data...")
registered = load_registered_face()
print(f"Loaded {len(registered)} face samples. Starting camera...")

cap = cv2.VideoCapture(0)
THRESHOLD = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = get_face_embedding(frame)

    if result is not None:
        embedding, (x, y, w, h) = result
        similarities = cosine_similarity(embedding, registered)
        score = similarities.max()

        if score >= THRESHOLD:
            label = "Access Granted - Fakhriddin"
            color = (0, 255, 0)
        else:
            label = f"Access Denied"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Match: {score:.0%}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Banking App - Face Login", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()