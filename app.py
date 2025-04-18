import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = os.getenv('SECRET_KEY')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Dummy emotion detection function
def predict_emotion(image_path):
    # Load a dummy emotion, replace this with your model prediction
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
    return np.random.choice(emotions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion = predict_emotion(filepath)
        return render_template('result.html', filename=filename, emotion=emotion)

    else:
        flash('Invalid file format. Please upload a jpg or png.')
        return redirect(url_for('index'))

# Route for viewing uploaded file
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=True)
