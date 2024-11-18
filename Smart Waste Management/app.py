
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained CNN model
model = tf.keras.models.load_model('model/waste_classifier.h5')

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Waste categories
categories = ['trash', 'plastic', 'paper', 'metal', 'glass', 'cardboard']

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).resize((150, 150))  # Resize image to 150x150
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict waste category
def predict_waste(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return categories[predicted_class_index]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Predict waste category
        category = predict_waste(file_path)
        
        # Provide waste handling solutions
        disposal_tips = {
            'glass': 'Recycle glass items in the glass recycling bin.',
            'paper': 'Recycle paper in the paper recycling bin.',
            'cardboard': 'Recycle cardboard in the cardboard recycling bin.',
            'metal': 'Recycle metal items in the metal recycling bin.',
            'plastic': 'Recycle plastic in the plastic recycling bin.',
            'trash': 'Dispose of general trash in the general waste bin.'
        }

        return jsonify({
            'category': category,
            'disposal_tip': disposal_tips[category],
            'file_path': file_path
        })

if __name__ == '__main__':
    app.run(debug=True)
