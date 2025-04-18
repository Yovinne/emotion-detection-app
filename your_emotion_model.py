# your_emotion_model.py
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')  # Your trained emotion model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))  # Resize to model input size
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)  # Add channel dimension if needed
    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion
