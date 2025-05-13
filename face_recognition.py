import cv2
import numpy as np
from siamese_model import get_embedding_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yüzü tespit et ve embedding çıkar
def extract_face_embedding(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        embedding = model.predict(face)[0]
        return embedding
    return None
