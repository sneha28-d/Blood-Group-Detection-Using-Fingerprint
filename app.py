from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

# Model path (change to .keras if needed)
MODEL_PATH = 'model_resnet_blood_group_detection.h5'

# Load the pre-trained model (supports .h5 and .keras)
def load_keras_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in ['.h5', '.keras']:
        return load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

model = load_keras_model(MODEL_PATH)

# Define the class labels
labels = {0: 'A+', 1: 'A-', 2: 'AB+', 3: 'AB-', 4: 'B+', 5: 'B-', 6: 'O+', 7: 'O-'}

# Ensure 'static/' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Preprocess the image to match model input shape
def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)  # Load image
    x = image.img_to_array(img)  # Convert image to array
    x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input
    x = preprocess_input(x)  # Preprocess input
    return x

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected")

        # Save the file temporarily with a unique name
        import uuid
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join('static', filename)
        file.save(file_path)

        print("✅ File uploaded:", file_path)  # Debugging log

        # Preprocess the image
        img = preprocess_image(file_path, target_size=(256, 256))
        print("✅ Image shape after preprocessing:", img.shape)  # Debugging log

        # Predict the blood group
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)  # Get the predicted class index
        blood_group = labels[predicted_class]
        confidence = round(float(prediction[0][predicted_class]) * 100, 2)  # Confidence level as float, rounded
        
        print("✅ Model prediction:", prediction)  # Debugging log
        print("✅ Predicted Blood Group:", blood_group)  # Debugging log

        # Redirect to results page with prediction
        return redirect(url_for('show_result', blood_group=blood_group, confidence=confidence, image_path=filename))

    return render_template('index.html')

@app.route('/result')
def show_result():
    # Get the predicted blood group, confidence, and image path from the URL parameters
    blood_group = request.args.get('blood_group')
    confidence = request.args.get('confidence')
    image_path = request.args.get('image_path')

    return render_template('result.html', blood_group=blood_group, confidence=confidence, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)