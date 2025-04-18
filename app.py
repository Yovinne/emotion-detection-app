from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Initialize the app
app = Flask(__name__)

# Load the model (make sure the .h5 file is in the same directory or provide correct path)
model = tf.keras.models.load_model('emotion_detection_model.keras')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No selected image.")

    # Preprocess image
    img = Image.open(file).convert('L')  # grayscale
    img = img.resize((48, 48))  # Resize to match model input
    img_array = np.array(img)
    img_array = img_array.reshape(1, 48, 48, 1) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = emotion_labels[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
