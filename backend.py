from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import os
import sqlite3
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define Upload Folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS clothes (id INTEGER PRIMARY KEY, user_id INTEGER, filename TEXT, category TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Helper function to classify image
def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    categories = ['tops', 'bottoms', 'shoes', 'accessories', 'jackets']
    return categories[np.argmax(predictions)]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    category = classify_image(file_path)
    return jsonify({'filename': filename, 'category': category})

if __name__ == '__main__':
    app.run(debug=True)
